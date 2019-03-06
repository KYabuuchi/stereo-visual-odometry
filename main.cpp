#include "feature.hpp"
#include "load.hpp"
#include "map.hpp"
#include "params.hpp"
#include "tracking.hpp"
#include "viewer.hpp"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    std::string file_path = "../data/VGA_100MM_FRONT";
    if (argc == 2)
        file_path = argv[1];

    Feature feature;
    Viewer viewer;
    Loader loader(file_path);

    std::vector<std::shared_ptr<MapPoint>> mappoints;
    cv::Mat cur_left_image, cur_right_image;
    cv::Mat pre_left_image, pre_right_image;

    int key = -1;
    bool initialize_done = false;

    // camera pose
    cv::Mat1f Tcw(cv::Mat1f::eye(4, 4));

    // Main Loop
    while (key != 'q') {

        // 画像取得
        if (not loader.load(cur_left_image, cur_right_image)) {
            // reset
            std::cout << "reset" << std::endl;
            loader.reset();
            viewer.reset();
            initialize_done = false;
            pre_left_image.release();
            pre_right_image.release();
            Tcw = cv::Mat1f::eye(4, 4);
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
        cv::Mat1f Tr = calcPose(mappoints);

        // 三角測量
        triangulate(mappoints);

        // Scaling
        float scale = calcScale(mappoints, Tr.colRange(0, 3).rowRange(0, 3));
        scaleTranslation(Tr, scale);
        if (scale > 0.2f) {
            std::cout << "(WARNING: too much movement) ";
        }
        std::cout << "(scale) " << scale << std::endl;

        // Integrate
        Tcw *= Tr;

        // 描画
        viewer.update({pre_left_image, pre_right_image, cur_left_image, cur_right_image}, Tcw, mappoints);
        std::cout << "(Tcw)\n"
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
        // key = viewer.waitKeyOnce();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    std::cout << "shut down" << std::endl;
}