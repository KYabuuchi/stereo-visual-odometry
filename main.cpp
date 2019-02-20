#include "feature.hpp"
#include "params.hpp"
#include "util.hpp"
#include "viewer.hpp"
#include <opencv2/opencv.hpp>

// 少なくともCL,CRがあることを保証する
class MapPoint
{
public:
    MapPoint(cv::Mat descriptor, cv::Point2f cur_left, cv::Point2f cur_right)
        : m_descriptor(descriptor), m_cur_left(cur_left), m_cur_right(cur_right) {}

    const cv::Mat m_descriptor;
    cv::Mat1f m_cur_left, m_cur_right, m_pre_left, m_pre_right;

    void update()
    {
        m_cur_left.copyTo(m_pre_left);
        m_cur_right.copyTo(m_pre_right);
    }

private:
};

int main()
{
    // 初期化
    Viewer::init();
    Feature::init();

    cv::Mat src1, src2;
    if (not readImage(1, src1, src2))
        return -1;

    std::vector<MapPoint> mappoints;
    mappoints.reserve(500 * 2);

    std::vector<cv::Point2f> keypoints1;
    std::vector<cv::Point2f> keypoints2;
    cv::Mat descriptor1, descriptor2;
    std::vector<cv::DMatch> matches;

    Feature::compute(src1, keypoints1, descriptor1);
    Feature::compute(src2, keypoints2, descriptor2);
    Feature::matching(descriptor1, descriptor2, matches);
    for (size_t i = 0; i < matches.size(); i++) {
        int query = matches.at(i).queryIdx;
        int train = matches.at(i).trainIdx;
        MapPoint mp(descriptor1.row(query), keypoints1.at(query), keypoints2.at(train));
        mappoints.push_back(mp);
    }
    std::cout << matches.size() << " mappoint is created" << std::endl;

    cv::Mat show, merge1, merge2;
    cv::imshow(Params::WINDOW_NAME, show);
    cv::waitKey(0);
}