#pragma once
#include "map.hpp"
#include <memory>
#include <opencv2/opencv.hpp>


bool readImage(int file_num, cv::Mat& src1, cv::Mat& src2);

// Epipolar方程式を導出して姿勢を計算する
cv::Mat calcPose(const std::vector<MapPointPtr>& mappoints);

// 3角測量可能な点について，三角測量する
int triangulate(std::vector<MapPointPtr>& mappoints);

// 3次元座標の移動量から並進ベクトルの大きさを計算する
float calcScale(const std::vector<MapPointPtr>& mappoints, const cv::Mat1f& R);
