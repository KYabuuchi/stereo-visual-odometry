#pragma once
#include <opencv2/opencv.hpp>

namespace Feature
{
void init();

void compute(cv::Mat image, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptors);

void matching(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches);

}  // namespace Feature