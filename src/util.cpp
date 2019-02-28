#include "util.hpp"
#include "params.hpp"

bool readImage(int file_num, cv::Mat& src1, cv::Mat& src2)
{
    //if (file_num > Params::MAX_FILE_NUM)
    //    file_num = Params::MAX_FILE_NUM;
    std::string file_path = "../data/VGA10CM/ZED_image" + std::to_string(file_num) + ".png";

    cv::Mat src = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    if (src.empty())
        return false;

    src1 = src.colRange(0, src.cols / 2);
    src2 = src.colRange(src.cols / 2, src.cols);
    return true;
}

cv::Mat calcPose(const std::vector<MapPointPtr>& mappoints)
{
    cv::Mat T = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    std::vector<cv::Point2f> cur_left, pre_left;
    for (const MapPointPtr mp : mappoints) {
        if (not mp->motionEstimatable())
            continue;
        cur_left.push_back(mp->curLeft());
        pre_left.push_back(mp->preLeft());
    }

    if (cur_left.size() < 5)
        return T;

    cv::Mat t, R, E, mask;
    E = cv::findEssentialMat(cur_left, pre_left, Params::ZED_INTRINSIC, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, cur_left, pre_left, Params::ZED_INTRINSIC, R, t, mask);
    R.copyTo(T.colRange(0, 3).rowRange(0, 3));
    t.copyTo(T.rowRange(0, 3).col(3));

    return T;
}

float calcScale(const std::vector<MapPointPtr>& mappoints, const cv::Mat1f& R)
{
    cv::Point3f pre(0, 0, 0);
    cv::Point3f cur(0, 0, 0);
    int num = 0;
    for (const MapPointPtr mp : mappoints) {
        if (not mp->scaleEstimatable())
            continue;
        cur += mp->curStruct();
        pre += mp->preStruct();
        num++;
    }

    if (num > 0) {
        cv::Mat pre_point(pre / num);
        cv::Mat cur_point(cur / num);
        return static_cast<float>(cv::norm(pre_point - R * cur_point));
    }
    return 0.0f;
}

size_t triangulate(std::vector<MapPointPtr>& mappoints)
{
    std::vector<cv::Point2f> cur_left, cur_right;
    std::vector<MapPointPtr> triangulatable_points;
    for (const MapPointPtr mp : mappoints) {
        if (not mp->triangulatable())
            continue;
        cur_left.push_back(mp->curLeft());
        cur_right.push_back(mp->curRight());
        triangulatable_points.push_back(mp);
    }
    cv::Mat homo3d, tmp;
    cv::triangulatePoints(
        Params::ZED_PERSPECTIVE_LEFT,
        Params::ZED_PERSPECTIVE_RIGHT,
        cur_left, cur_right, homo3d);  // 4xN(1ch)

    cv::convertPointsFromHomogeneous(homo3d.t(), tmp);  // Nx4(1ch) -> Nx1(3ch)
    tmp = tmp.reshape(1);                               // Nx3(4ch) -> Nx3(1ch)
    tmp = tmp.t();                                      // Nx3(1ch) -> 3xN(1ch)
    for (int i = 0; i < tmp.cols; i++) {
        triangulatable_points.at(i)->setCurStruct(cv::Point3f(tmp.col(i)));
    }

    return triangulatable_points.size();
}


size_t initializeMapPoints(
    std::vector<MapPointPtr>& mappoints,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& left_descriptors,
    const cv::Mat&,
    const std::vector<cv::Point2f>& left_keypoints,
    const std::vector<cv::Point2f>& right_keypoints)
{
    mappoints.clear();
    for (size_t i = 0; i < matches.size(); i++) {
        int query = matches.at(i).queryIdx;
        int train = matches.at(i).trainIdx;
        MapPointPtr mp = std::make_shared<MapPoint>(
            left_descriptors.row(query),
            left_keypoints.at(query),
            right_keypoints.at(train));
        mappoints.push_back(mp);
    }
    return mappoints.size();
}

cv::Mat concatenateDescriptors(const std::vector<MapPointPtr> mappoints, const Feature& feature)
{
    cv::Mat ref_descriptors = cv::Mat(0, feature.descriptorSize(), feature.descriptorType());
    for (const MapPointPtr& mp : mappoints) {
        cv::vconcat(ref_descriptors, mp->m_descriptor, ref_descriptors);
    }
    return ref_descriptors;
}