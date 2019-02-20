#include "params.hpp"

namespace Params
{
const int MAX_FILE_NUM = 8;

const std::string WINDOW_NAME = "window";

const cv::Mat1f ZED_INTRINSIC = (cv::Mat_<float>(3, 3) << 350, 0, 336, 0, 350, 336, 0, 1);
const cv::Mat1f ZED_EXTRINSIC = (cv::Mat_<float>(3, 4) << 1, 0, 0, 120, 0, 1, 0, 0, 0, 0, 1, 0);

}  // namespace Params