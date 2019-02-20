#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace Params
{
extern const int MAX_FILE_NUM;
extern const std::string WINDOW_NAME;

extern const cv::Mat1f ZED_INTRINSIC;
extern const cv::Mat1f ZED_EXTRINSIC;
}  // namespace Params