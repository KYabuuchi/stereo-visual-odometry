#pragma once
#include "feature.hpp"
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

// 一番最初のMapPoint作り
int initializeMapPoints(
    std::vector<MapPointPtr>& mappoints,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& left_descriptors,
    const cv::Mat& right_descriptors,
    const std::vector<cv::Point2f>& left_keypoints,
    const std::vector<cv::Point2f>& right_keypoints);

cv::Mat concatenateDescriptors(
    const std::vector<MapPointPtr> mappoints,
    const Feature& feature);