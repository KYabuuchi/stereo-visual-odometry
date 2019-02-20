#include "feature.hpp"
#include "params.hpp"
#include "util.hpp"
#include "viewer.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    // 初期化
    Viewer::init();
    Feature::init();

    cv::Mat src1, src2;
    if (not readImage(1, src1, src2))
        return -1;

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptor1, descriptor2;

    Feature::compute(src1, keypoints1, descriptor1);
    Feature::compute(src2, keypoints2, descriptor2);


    Viewer::show(src1, src2, src1, src2);
    cv::waitKey(0);
}