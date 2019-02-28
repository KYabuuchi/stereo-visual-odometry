#pragma once
#include <opencv2/opencv.hpp>

class Feature
{
public:
    Feature();

    void compute(cv::Mat image, std::vector<cv::Point2f>& keypoint, cv::Mat& descriptors) const;

    int descriptorSize() const;
    int descriptorType() const;

    void matching(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& matches,
        const float ratio = 0.6f) const;

private:
    cv::Ptr<cv::FeatureDetector> m_detector;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;
};
