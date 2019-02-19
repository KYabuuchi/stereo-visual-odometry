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

bool showImage(cv::Mat cur_left_image, cv::Mat cur_right_image, cv::Mat pre_left_image, cv::Mat pre_right_image)
{
    cv::Mat merge1, merge2;
    cv::hconcat(cur_left_image, cur_right_image, merge1);
    cv::hconcat(pre_left_image, pre_right_image, merge2);
    cv::vconcat(merge1, merge2, merge2);
    cv::imshow(Params::WINDOW_NAME, merge2);

    return true;
}