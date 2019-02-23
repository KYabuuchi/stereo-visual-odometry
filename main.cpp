#include "feature.hpp"
#include "params.hpp"
#include "util.hpp"
#include "viewer.hpp"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

typedef std::shared_ptr<MapPoint> MapPointPtr;

int main()
{
    // 初期化
    Viewer::init();
    Feature::init();

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

        // Epipolar
        cv::Mat T;
        {
            std::vector<cv::Point2f> cur_left, pre_left;
            for (const MapPointPtr mp : mappoints) {
                if (not mp->motionEstimatable())
                    continue;
                cur_left.push_back(mp->curLeft());
                pre_left.push_back(mp->preLeft());
            }
            T = calcPose(cur_left, pre_left);
        }

        // 三角測量
        {
            std::vector<cv::Point2f> cur_left, cur_right;
            for (const MapPointPtr mp : mappoints) {
                if (not mp->triangulatable())
                    continue;
                cur_left.push_back(mp->curLeft());
                cur_right.push_back(mp->curRight());
            }
            cv::Mat homo3d, tmp;
            cv::triangulatePoints(
                Params::ZED_PERSPECTIVE_LEFT,
                Params::ZED_PERSPECTIVE_RIGHT,
                cur_left, cur_right, homo3d);                   // 4xN(1ch)
            cv::convertPointsFromHomogeneous(homo3d.t(), tmp);  // Nx4(1ch) -> Nx1(3ch)
            cv::convertPointsToHomogeneous(tmp, tmp);           // Nx1(3ch) -> Nx1(4ch)
            tmp = tmp.reshape(1);                               // Nx1(4ch) -> Nx4(1ch)
            tmp = tmp.t();                                      // Nx4(1ch) -> 4xN(1ch)
            tmp = tmp.rowRange(0, 3);                           // 4xN(1ch) -> 3xN(1ch)
            //std::cout << tmp.col(0) << std::endl;
        }

        // スケールの計算
        float scale = 1;
        {
            std::vector<cv::Point3f> pre_struct, cur_struct;
            for (const MapPointPtr mp : mappoints) {
                if (not mp->scaleEstimatable())
                    continue;
                cur_struct.push_back(mp->curStruct());
                pre_struct.push_back(mp->preStruct());
            }
        }
        std::cout << "\nTranslation " << scale << "\n"
                  << T << "\n"
                  << std::endl;

        // 描画
        {
            cv::Mat show, merge1, merge2;
            cv::hconcat(pre_left_image, pre_right_image, merge1);
            cv::hconcat(cur_left_image, cur_right_image, merge2);
            cv::vconcat(merge1, merge2, show);

            const cv::Size size = cur_left_image.size();
            const cv::Point2f OFFSET_PL(0, 0);
            const cv::Point2f OFFSET_PR(size.width, 0);
            const cv::Point2f OFFSET_CL(0, size.height);
            const cv::Point2f OFFSET_CR(size.width, size.height);

            for (const MapPointPtr p : mappoints) {
                if (p->enable(PL))
                    cv::circle(show, p->preLeft() + OFFSET_CL, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(PR))
                    cv::circle(show, p->preRight() + OFFSET_PR, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(CL))
                    cv::circle(show, p->curLeft() + OFFSET_CL, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(CR))
                    cv::circle(show, p->curRight() + OFFSET_CL, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);

                if (p->triangulatable())
                    cv::line(show, p->curRight() + OFFSET_CL, p->curLeft() + OFFSET_CL, CV_RGB(255, 255, 0), 1, cv::LineTypes::LINE_AA);
                if (p->motionEstimatable())
                    cv::line(show, p->preLeft() + OFFSET_CL, p->curLeft() + OFFSET_CL, CV_RGB(0, 255, 255), 1, cv::LineTypes::LINE_AA);
            }
            cv::imshow(Params::WINDOW_NAME, show);
        }

        // 更新
        for (MapPointPtr mp : mappoints) {
            mp->update();
        }
        cur_left_image.copyTo(pre_left_image);
        cur_right_image.copyTo(pre_right_image);

        key = cv::waitKey(0);
    }

    std::cout << "shut down" << std::endl;
}