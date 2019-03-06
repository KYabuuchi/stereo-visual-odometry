#pragma once
#include <opencv2/opencv.hpp>

// 3xNの実数行列の平均を求める
cv::Scalar robustMean(const cv::Mat& mat_1ch, bool debug = false)
{
    cv::Scalar mean, sigma;
    cv::Mat mat_3ch = cv::Mat(mat_1ch.t()).reshape(3);
    cv::meanStdDev(mat_3ch, mean, sigma);
    if (debug) {
        std::cout << "before mean : " << mean << std::endl;
        std::cout << "before simga: " << sigma << std::endl;
    }

    cv::Mat mask(mat_3ch.rows, 1, CV_8UC1);
    auto func = [&mask, mean, sigma](cv::Vec3f& p, const int* position) -> void {
        bool enable = true;
        for (int i = 0; i < 3; i++) {
            float diff = (p[i] - static_cast<float>(mean[i]));
            enable &= std::abs(diff) < sigma[i] * 0.8;
        }
        mask.at<unsigned char>(*position) = (enable) ? 1 : 0;
    };
    mat_3ch.forEach<cv::Vec3f>(func);

    cv::meanStdDev(mat_3ch, mean, sigma, mask);
    if (debug) {
        std::cout << "after mean  : " << mean << std::endl;
        std::cout << "after simga : " << sigma << std::endl;
    }
    return mean;
}