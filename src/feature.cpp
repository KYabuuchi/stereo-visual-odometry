#include "feature.hpp"
#include <iostream>

namespace Feature
{
cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorMatcher> matcher;


int descriptorSize() { return detector->descriptorSize(); }
int descriptorType() { return detector->descriptorType(); }

void init()
{
    // TODO: Paramsで色々変えられるようにする
    detector = cv::AKAZE::create();
    matcher = cv::BFMatcher::create(detector->defaultNorm());
    std::cout << "Feature type: " << detector->getDefaultName() << std::endl;
}

void compute(cv::Mat image, std::vector<cv::Point2f>& points, cv::Mat& descriptors)
{
    std::vector<cv::KeyPoint> keypoints;
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

// mutual matching(少ないものは1にセットすると良い)
// 1 -> 2 -> 1
void matching(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches)
{
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 1, cv::noArray());

    std::vector<cv::DMatch> oneway_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].empty()) {
            continue;
        }
        oneway_matches.push_back(knn_matches[i][0]);
    }

    knn_matches.clear();  // NOTE: 必須
    matcher->knnMatch(descriptors2, descriptors1, knn_matches, 1, cv::noArray());
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].empty()) {
            continue;
        }

        cv::DMatch backward = knn_matches[i][0];
        int query1 = backward.queryIdx;
        int train1 = backward.trainIdx;

        cv::DMatch forward = oneway_matches[train1];
        int query2 = forward.queryIdx;
        int train2 = forward.trainIdx;

        if (query1 == train2) {
            float distance = forward.distance;
            matches.push_back(cv::DMatch(query2, query1, distance));
        }
    }
}

}  // namespace Feature