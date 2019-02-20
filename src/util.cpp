#include "util.hpp"
#include "params.hpp"

bool readImage(int file_num, cv::Mat& src1, cv::Mat& src2)
{
    if (file_num > Params::MAX_FILE_NUM)
        file_num = Params::MAX_FILE_NUM;
    std::string file_path = "../data/VGA10CM/ZED_image" + std::to_string(file_num) + ".png";

    cv::Mat src = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    if (src.empty())
        return false;

    src1 = src.colRange(0, src.cols / 2);
    src2 = src.colRange(src.cols / 2, src.cols);
    return true;
}
