#include "tracking.hpp"
#include "params.hpp"

cv::Mat calcPose(const std::vector<MapPointPtr>& mappoints)
{
    cv::Mat1f T = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
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
    cv::Mat1f pre = cv::Mat1f(cv::Mat1f::zeros(3, 0));
    cv::Mat1f cur = cv::Mat1f(cv::Mat1f::zeros(3, 0));
    for (const MapPointPtr mp : mappoints) {
        if (not mp->scaleEstimatable())
            continue;
        cv::hconcat(cur, mp->curStruct(), cur);
        cv::hconcat(pre, mp->preStruct(), pre);
    }

    if (cur.cols > 0) {
        cv::Mat1f diff = pre - R * cur;
        diff = diff.t();
        cv::Mat3f tmp = diff.reshape(3);
        cv::Scalar scalar = cv::mean(tmp);
        return static_cast<float>(cv::norm(scalar));
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
        cv::Point3f p = cv::Point3f(tmp.col(i));
        if (p.z < 0) {
            std::cerr << "negative depth" << std::endl;
            continue;
        }
        triangulatable_points.at(i)->setCurStruct(p);
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

void scaleTranslation(cv::Mat1f& T, float scale)
{
    cv::Mat1f translation = T.rowRange(0, 3).col(3);
    float norm = static_cast<float>(cv::norm(translation));
    if (norm < 1e-6f)
        return;
    translation = translation / norm * scale;
    translation.copyTo(T.rowRange(0, 3).col(3));
}