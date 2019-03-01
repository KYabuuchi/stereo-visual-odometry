#include "feature.hpp"
#include "load.hpp"
#include "map.hpp"
#include "params.hpp"
#include "tracking.hpp"
#include "viewer.hpp"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

const std::string path_to_data = "../data/VGA10CM";

int main()
{
    Feature feature;
    Viewer viewer;
    Loader loader(path_to_data);

    std::vector<std::shared_ptr<MapPoint>> mappoints;
    mappoints.reserve(500 * 2);

    cv::Mat cur_left_image, cur_right_image;
    cv::Mat pre_left_image, pre_right_image;

    int key = -1;
    bool initialize_done = false;

    // Main Loop
    while (key != 'q') {

        // 画像取得
        if (not loader.load(cur_left_image, cur_right_image)) {
            std::cout << "loop" << std::endl;
            loader.reset();
            viewer.reset();
            initialize_done = false;
            pre_left_image.release();
            pre_right_image.release();
            continue;
        }

        std::vector<cv::Point2f> left_keypoints, right_keypoints;
        cv::Mat left_descriptors, right_descriptors;
        std::vector<cv::DMatch> matches;

        // 特徴点抽出
        feature.compute(cur_left_image, left_keypoints, left_descriptors);
        feature.compute(cur_right_image, right_keypoints, right_descriptors);

        if (not initialize_done) {
            // 初期化
            feature.matching(left_descriptors, right_descriptors, matches);
            initializeMapPoints(mappoints, matches, left_descriptors, right_descriptors, left_keypoints, right_keypoints);
            initialize_done = true;
        } else {
            // CL-PL対応
            cv::Mat ref_descriptors = concatenateDescriptors(mappoints, feature);
            feature.matching(left_descriptors, ref_descriptors, matches);

            // CLを追加する
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
                    mappoints.push_back(std::make_shared<MapPoint>(left_descriptors.row(static_cast<int>(i)), left_keypoints.at(static_cast<int>(i))));
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

        // 描画
        viewer.update({pre_left_image, pre_right_image, cur_left_image, cur_right_image}, mappoints);
        std::cout << "\nPose" << scale << "\n"
                  << Tcw << "\n"
                  << std::endl;

        // 更新
        for (MapPointPtr mp : mappoints) {
            mp->update();
        }
        pre_left_image = std::move(cur_left_image);
        pre_right_image = std::move(cur_right_image);

        // wait
        key = viewer.waitKeyEver();
    }

    viewer.stop();
    std::cout << "shut down" << std::endl;
}