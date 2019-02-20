#pragma once
#include <opencv2/opencv.hpp>

namespace Feature
{
void init();

void compute(cv::Mat image, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptors);

void matching(
    const std::vector<cv::KeyPoint> keypoint1, cv::Mat descriptors1,
    const std::vector<cv::KeyPoint> keypoint2, cv::Mat descriptors2,
    std::vector<cv::DMatch> matches);

}  // namespace Feature