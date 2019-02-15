#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat src = cv::imread("../data/image.png", cv::IMREAD_UNCHANGED);
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::waitKey(0);
}