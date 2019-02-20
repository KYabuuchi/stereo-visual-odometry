#pragma once
#include <opencv2/opencv.hpp>

namespace Viewer
{
void init();
bool show(cv::Mat cur_left_image,
    cv::Mat cur_right_image,
    cv::Mat pre_left_image,
    cv::Mat pre_right_image);

}  // namespace Viewer