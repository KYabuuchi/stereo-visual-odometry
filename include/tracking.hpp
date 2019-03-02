#pragma once
#include "feature.hpp"
#include "map.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

// Epipolar方程式を導出して姿勢を計算する
cv::Mat calcPose(const std::vector<MapPointPtr>& mappoints);

// 3角測量可能な点について，三角測量する
size_t triangulate(std::vector<MapPointPtr>& mappoints);

// 3次元座標の移動量から並進ベクトルの大きさを計算する
float calcScale(const std::vector<MapPointPtr>& mappoints, const cv::Mat1f& R);

// 一番最初のMapPoint作り
size_t initializeMapPoints(
    std::vector<MapPointPtr>& mappoints,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& left_descriptors,
    const cv::Mat& right_descriptors,
    const std::vector<cv::Point2f>& left_keypoints,
    const std::vector<cv::Point2f>& right_keypoints);

// 特徴量記述子を重ねる
cv::Mat concatenateDescriptors(
    const std::vector<MapPointPtr> mappoints,
    const Feature& feature);

// 並進ベクトルをscaling
void scaleTranslation(cv::Mat1f& T, float scale);