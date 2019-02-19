#include "params.hpp"
#include "util.hpp"
#include <opencv2/opencv.hpp>

constexpr int MAX_FILE_NUM = 8;
const std::string WINDOW_NAME = "window";

int main()
{
    cv::Mat src1, src2;
    if (not readImage(1, src1, src2))
        return -1;
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    showImage(src1, src2, src1, src2);
    cv::waitKey(0);
}