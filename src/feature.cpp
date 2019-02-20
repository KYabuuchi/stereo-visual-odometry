#include "feature.hpp"
#include <iostream>

namespace Feature
{
cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorMatcher> matcher;

void init()
{
    // NOTE: Paramsで色々変えられるようにする
    detector = cv::AKAZE::create();
    matcher = cv::BFMatcher::create(detector->defaultNorm());
    std::cout << "Feature type: " << detector->getDefaultName() << std::endl;
}

void compute(cv::Mat image, std::vector<cv::KeyPoint>& keypoint, cv::Mat& descriptors)
{
    detector->detectAndCompute(image, cv::noArray(), keypoint, descriptors);
}

void matching(
    const std::vector<cv::KeyPoint> keypoint1, cv::Mat descriptors1,
    const std::vector<cv::KeyPoint> keypoint2, cv::Mat descriptors2,
    std::vector<cv::DMatch> matches)
{
}

}  // namespace Feature