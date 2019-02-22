#include "feature.hpp"
#include "params.hpp"
#include "util.hpp"
#include "viewer.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

const int PL = 0;
const int PR = 1;
const int CL = 2;
const int CR = 3;
const int PRE = 4;
const int CUR = 5;

// 少なくともCLがあることを保証する
class MapPoint
{
public:
    // 特徴量記述子とCL
    MapPoint(cv::Mat descriptor, cv::Point2f cur_left)
        : m_descriptor(descriptor), m_feature{cv::Point2f(-1, -1), cv::Point2f(-1, -1), cur_left, cv::Point2f(-1, -1)},
          m_cur_struct(cv::Point3f(-1, -1, -1)), m_pre_struct(cv::Point3f(-1, -1, -1)) {}

    // 特徴量記述子とCLとCR
    MapPoint(cv::Mat descriptor, cv::Point2f cur_left, cv::Point2f cur_right)
        : m_descriptor(descriptor), m_feature{cv::Point2f(-1, -1), cv::Point2f(-1, -1), cur_left, cur_right},
          m_cur_struct(cv::Point3f(-1, -1, -1)), m_pre_struct(cv::Point3f(-1, -1, -1)) {}

    // 時間方向更新
    void update()
    {
        m_feature.at(PR) = m_feature.at(CR);
        m_feature.at(PL) = m_feature.at(CL);
        m_pre_struct = m_cur_struct;
        m_feature.at(CR) = cv::Point2f(-1, -1);
        m_feature.at(CL) = cv::Point2f(-1, -1);
        m_cur_struct = cv::Point3f(-1, -1, -1);
    }

    // 姿勢推定が可能か否か
    bool motionEstimatable() const { return enable(PL) and enable(CL); }
    // 三角測量が可能か否か
    bool triangulatable() const { return enable(CL) and enable(CR); }
    // スケール推定が可能か否か
    bool scaleEstimatable() const { return enable(PRE) and enable(CUR); }

    bool enable(int id) const
    {
        if (id == PRE)
            return m_pre_struct.x >= 0;
        if (id == CUR)
            return m_cur_struct.x >= 0;
        return m_feature.at(id).x >= 0;
    }

    void setCurLeft(cv::Point2f kp) { m_feature.at(CL) = kp; }
    void setCurRight(cv::Point2f kp) { m_feature.at(CR) = kp; }

    cv::Point2f preLeft() const { return m_feature.at(PL); }
    cv::Point2f preRight() const { return m_feature.at(PR); }
    cv::Point2f curLeft() const { return m_feature.at(CL); }
    cv::Point2f curRight() const { return m_feature.at(CR); }

    // 特徴量記述子
    const cv::Mat m_descriptor;

private:
    // 現在・過去x左・右の特徴点座標[pixel]
    std::array<cv::Point2f, 4> m_feature;
    // 現在・過去の3次元点[m]
    cv::Point3f m_cur_struct, m_pre_struct;
};

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
    std::vector<MapPoint*> mappoints;
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
            MapPoint* mp = new MapPoint(left_descriptors.row(query), left_keypoints.at(query), right_keypoints.at(train));
            mappoints.push_back(mp);
        }
    }

    // 更新
    for (MapPoint* mp : mappoints) {
        mp->update();
    }
    cur_left_image.copyTo(pre_left_image);
    cur_right_image.copyTo(pre_right_image);

    // Main Loop
    while (1) {
        std::cout << " mappoints has " << mappoints.size() << " elements" << std::endl;

        // 画像取得
        if (not readImage(2, cur_left_image, cur_right_image)) {
            std::cout << "cannot read image" << std::endl;
            return -1;
        }
        {
            std::vector<cv::Point2f> left_keypoints, right_keypoints;
            cv::Mat left_descriptors, right_descriptors;
            std::vector<cv::DMatch> matches;

            // 特徴点抽出
            Feature::compute(cur_left_image, left_keypoints, left_descriptors);
            Feature::compute(cur_right_image, right_keypoints, right_descriptors);
            std::cout << "1" << std::endl;

            // 時間方向対応
            cv::Mat pre_descriptors = cv::Mat(0, Feature::descriptorSize(), Feature::descriptorType());
            for (const MapPoint* mp : mappoints) {
                cv::vconcat(pre_descriptors, mp->m_descriptor, pre_descriptors);
            }
            Feature::matching(left_descriptors, pre_descriptors, matches);

            // 対応のあるCLを追加する
            std::vector<bool> already_pushed(left_keypoints.size(), false);
            for (const cv::DMatch& match : matches) {
                int query = match.queryIdx;
                int train = match.trainIdx;
                mappoints.at(train)->setCurLeft(left_keypoints.at(query));
                already_pushed.at(query) = true;
            }

            // 追加のされなかったmappointを消す
            for (std::vector<MapPoint*>::iterator it = mappoints.begin(); it != mappoints.end();) {
                if ((*it)->enable(CL)) {
                    it++;
                    continue;
                }

                // NOTE: なら，shared_ptrを使えばいいのに
                delete *it;
                it = mappoints.erase(it);
            }

            // CLしかないものも追加
            for (size_t i = 0; i < left_keypoints.size(); i++) {
                if (already_pushed.at(i))
                    continue;
                MapPoint* mp = new MapPoint(left_descriptors.row(i), left_keypoints.at(i));
                mappoints.push_back(mp);
            }

            std::cout << "2" << std::endl;
            // 空間方向対応
            Feature::matching(left_descriptors, right_descriptors, matches);
        }

        // Epipolar

        // 三角測量

        // スケーリング

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

            for (const MapPoint* p : mappoints) {
                if (p->enable(PL))
                    cv::circle(show, p->preLeft() + OFFSET_PL, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(PR))
                    cv::circle(show, p->preRight() + OFFSET_PR, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->triangulatable()) {
                    cv::line(show, p->curRight() + OFFSET_CR, p->curLeft() + OFFSET_CR, CV_RGB(255, 0, 0), 1, cv::LineTypes::LINE_AA);
                }
                if (p->motionEstimatable())
                    cv::line(show, p->preLeft() + OFFSET_PL, p->curLeft() + OFFSET_CL, CV_RGB(255, 0, 0), 1, cv::LineTypes::LINE_AA);
                if (p->enable(CL))
                    cv::circle(show, p->curLeft() + OFFSET_CL, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(CR))
                    cv::circle(show, p->curRight() + OFFSET_CR, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
            }
            cv::imshow(Params::WINDOW_NAME, show);
            cv::waitKey(0);
        }

        // 更新
        for (MapPoint* mp : mappoints) {
            mp->update();
        }
        cur_left_image.copyTo(pre_left_image);
        cur_right_image.copyTo(pre_right_image);
    }
}