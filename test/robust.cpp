#include "util.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    const int N = 10;
    cv::Mat1f mat(3, N);

    // uniform distribution
    cv::randu(mat, -100.0f, 100.0f);

    // outlier
    for (int i = 0; i < N / 5; i++) {
        for (int j = 0; j < 3; j++)
            mat.at<float>(j, 5 * i) = 1000;
    }
    std::cout << mat << std::endl;
    std::cout << std::endl;

    robustMean(mat, true);
}