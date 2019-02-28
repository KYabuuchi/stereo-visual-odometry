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
    Feature feature;
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

        feature.compute(cur_left_image, left_keypoints, left_descriptors);
        feature.compute(cur_right_image, right_keypoints, right_descriptors);
        feature.matching(left_descriptors, right_descriptors, matches);

        initializeMapPoints(mappoints, matches, left_descriptors, right_descriptors, left_keypoints, right_keypoints);
        triangulate(mappoints);

        for (MapPointPtr mp : mappoints) {
            mp->update();
        }
        cur_left_image.copyTo(pre_left_image);
        cur_right_image.copyTo(pre_right_image);
    }


    int image_num = 2;
    int key = -1;

    // Main Loop
    while (key != 'q') {
        std::cout << "mappoints has " << mappoints.size() << " elements" << std::endl;

        // 画像取得
        if (not readImage(image_num++, cur_left_image, cur_right_image)) {
            std::cout << "cannot read image" << std::endl;

            // NOTE: 無理やりループ
            image_num = 2;
            continue;
        }

        {
            std::vector<cv::Point2f> left_keypoints, right_keypoints;
            cv::Mat left_descriptors, right_descriptors;
            std::vector<cv::DMatch> matches;

            // 特徴点抽出
            feature.compute(cur_left_image, left_keypoints, left_descriptors);
            feature.compute(cur_right_image, right_keypoints, right_descriptors);

            // CL-PL対応
            cv::Mat ref_descriptors = concatenateDescriptors(mappoints, feature);
            feature.matching(left_descriptors, ref_descriptors, matches);

            // 対応のあるCLを追加する
            std::vector<bool> already_pushed(left_keypoints.size(), false);
            for (const cv::DMatch& match : matches) {
                mappoints.at(match.trainIdx)->setCurLeft(left_keypoints.at(match.queryIdx));
                already_pushed.at(match.queryIdx) = true;
            }

            // CLを持たないmappointを消す
            for (std::vector<MapPointPtr>::iterator it = mappoints.begin(); it != mappoints.end();) {
                if ((*it)->enable(CL)) {
                    it++;
                    continue;
                }
                it = mappoints.erase(it);
            }

            // まだ追加されていない分をmappointを追加
            for (size_t i = 0; i < left_keypoints.size(); i++) {
                if (not already_pushed.at(i))
                    mappoints.push_back(std::make_shared<MapPoint>(left_descriptors.row(i), left_keypoints.at(i)));
            }

            // CR-CL対応
            ref_descriptors = concatenateDescriptors(mappoints, feature);
            feature.matching(right_descriptors, ref_descriptors, matches);
            for (const cv::DMatch& match : matches) {
                mappoints.at(match.trainIdx)->setCurRight(right_keypoints.at(match.queryIdx));
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
    }

    std::cout << "shut down" << std::endl;
}