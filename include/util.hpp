#pragma once
#include <opencv2/opencv.hpp>

bool readImage(int file_num, cv::Mat& src1, cv::Mat& src2);

bool showImage(cv::Mat cur_left_image, cv::Mat cur_right_image, cv::Mat pre_left_image, cv::Mat pre_right_image);