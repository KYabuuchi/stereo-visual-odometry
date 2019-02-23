#pragma once
#include <opencv2/opencv.hpp>

bool readImage(int file_num, cv::Mat& src1, cv::Mat& src2);

cv::Mat calcPose(
    const std::vector<cv::Point2f>& cur_left,
    const std::vector<cv::Point2f>& pre_left);

namespace
{
const int PL = 0;
const int PR = 1;
const int CL = 2;
const int CR = 3;
const int PRE = 4;
const int CUR = 5;
}  // namespace

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
