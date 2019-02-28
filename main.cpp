#include "feature.hpp"
#include "map.hpp"
#include "params.hpp"
#include "util.hpp"
#include "viewer.hpp"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

int main()
{
    Feature::init();
    Viewer viewer;

    cv::Mat cur_left_image, cur_right_image;
    cv::Mat pre_left_image, pre_right_image;

    if (not readImage(1, cur_left_image, cur_right_image))
        return -1;

    // 各視点での特徴点
    std::vector<std::shared_ptr<MapPoint>> mappoints;
    mappoints.reserve(500 * 2);

    {
        std::vector<cv::Point2f> left_keypoints, right_keypoints;
        cv::Mat left_descriptors, right_descriptors;
        std::vector<cv::DMatch> matches;

        Feature::compute(cur_left_image, left_keypoints, left_descriptors);
        Feature::compute(cur_right_image, right_keypoints, right_descriptors);
        Feature::matching(left_descriptors, right_descriptors, matches);
        for (size_t i = 0; i < matches.size(); i++) {
            int query = matches.at(i).queryIdx;
            int train = matches.at(i).trainIdx;
            MapPointPtr mp = std::make_shared<MapPoint>(left_descriptors.row(query), left_keypoints.at(query), right_keypoints.at(train));
            mappoints.push_back(mp);
        }
        triangulate(mappoints);
    }

    // 更新
    for (MapPointPtr mp : mappoints) {
        mp->update();
    }
    cur_left_image.copyTo(pre_left_image);
    cur_right_image.copyTo(pre_right_image);

    int image_num = 2;
    int key = -1;

    // Main Loop
    while (key != 'q') {
        std::cout << "mappoints has " << mappoints.size() << " elements" << std::endl;

        // 画像取得
        if (not readImage(image_num++, cur_left_image, cur_right_image)) {
            std::cout << "cannot read image" << std::endl;
            return -1;
        }

        {
            std::vector<cv::Point2f> left_keypoints, right_keypoints;
            cv::Mat left_descriptors, right_descriptors;
            std::vector<cv::DMatch> matches;

            // 特徴点抽出
            std::cout << "extract feature" << std::endl;
            Feature::compute(cur_left_image, left_keypoints, left_descriptors);
            Feature::compute(cur_right_image, right_keypoints, right_descriptors);

            // 時間方向対応
            std::cout << "time matching" << std::endl;
            cv::Mat ref_descriptors = cv::Mat(0, Feature::descriptorSize(), Feature::descriptorType());
            for (const MapPointPtr mp : mappoints) {
                cv::vconcat(ref_descriptors, mp->m_descriptor, ref_descriptors);
            }
            Feature::matching(left_descriptors, ref_descriptors, matches);

            // 対応のあるCLを追加する
            std::vector<bool> already_pushed(left_keypoints.size(), false);
            for (const cv::DMatch& match : matches) {
                int query = match.queryIdx;
                int train = match.trainIdx;
                mappoints.at(train)->setCurLeft(left_keypoints.at(query));
                already_pushed.at(query) = true;
            }
            // 追加のされなかったmappointを消す
            for (std::vector<MapPointPtr>::iterator it = mappoints.begin(); it != mappoints.end();) {
                if ((*it)->enable(CL)) {
                    it++;
                    continue;
                }
                it = mappoints.erase(it);
            }
            // CLしかないものもmappointを追加
            for (size_t i = 0; i < left_keypoints.size(); i++) {
                if (already_pushed.at(i))
                    continue;
                MapPointPtr mp = std::make_shared<MapPoint>(left_descriptors.row(i), left_keypoints.at(i));
                mappoints.push_back(mp);
            }

            std::cout << "space matching" << std::endl;
            // 特徴量記述子の再編成
            ref_descriptors = cv::Mat(0, Feature::descriptorSize(), Feature::descriptorType());
            for (const MapPointPtr mp : mappoints) {
                cv::vconcat(ref_descriptors, mp->m_descriptor, ref_descriptors);
            }
            // 空間方向対応
            Feature::matching(right_descriptors, ref_descriptors, matches);
            for (const cv::DMatch& match : matches) {
                int query = match.queryIdx;
                int train = match.trainIdx;
                mappoints.at(train)->setCurRight(right_keypoints.at(query));
            }
        }

        // Epipolar Equation
        cv::Mat Tcw = calcPose(mappoints);

        // 三角測量
        triangulate(mappoints);

        // スケールの計算
        float scale = calcScale(mappoints, Tcw.colRange(0, 3).rowRange(0, 3));

        std::cout << "\nPose" << scale << "\n"
                  << Tcw << "\n"
                  << std::endl;

        // 描画
        viewer.update({pre_left_image, pre_right_image, cur_left_image, cur_right_image}, mappoints);

        // 更新
        for (MapPointPtr mp : mappoints) {
            mp->update();
        }
        pre_left_image = std::move(cur_left_image);
        pre_right_image = std::move(cur_right_image);

        // wait
        key = viewer.waitKeyEver();
        std::cout << "key: " << key << std::endl;
    }

    std::cout << "shut down" << std::endl;
}