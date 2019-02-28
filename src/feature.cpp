#include "feature.hpp"
#include <iostream>


int Feature::descriptorSize() const { return m_detector->descriptorSize(); }
int Feature::descriptorType() const { return m_detector->descriptorType(); }

Feature::Feature()
    : m_detector(cv::AKAZE::create()),
      m_matcher(cv::BFMatcher::create(m_detector->defaultNorm()))
{
    // TODO: Paramsで色々変えられるようにする
    std::cout << "Feature type: " << m_detector->getDefaultName() << std::endl;
}

void Feature::compute(cv::Mat image, std::vector<cv::Point2f>& points, cv::Mat& descriptors) const
{
    std::vector<cv::KeyPoint> keypoints;
    m_detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

// mutual matching(少ない集合は1にセットすると良い)
// 1 -> 2 -> 1
void Feature::matching(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches,
    const float ratio) const
{
    matches.clear();
    std::vector<std::vector<cv::DMatch>> knn_matches;
    m_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2, cv::noArray());

    std::vector<cv::DMatch> oneway_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() < 2)
            continue;
        if (knn_matches[i][0].distance > ratio * knn_matches[i][1].distance)
            continue;
        oneway_matches.push_back(knn_matches[i][0]);
        matches.push_back(knn_matches[i][0]);
    }

    // NOTE: そうでもない
    //knn_matches.clear();
    //matcher->knnMatch(descriptors2, descriptors1, knn_matches, 2, cv::noArray());
    //for (size_t i = 0; i < knn_matches.size(); i++) {
    //    if (knn_matches[i].size() < 2)
    //        continue;
    //    if (knn_matches[i][0].distance > ratio * knn_matches[i][1].distance)
    //        continue;

    //    cv::DMatch backward = knn_matches[i][0];
    //    int query1 = backward.queryIdx;
    //    int train1 = backward.trainIdx;

    //    cv::DMatch forward = oneway_matches[train1];
    //    int query2 = forward.queryIdx;
    //    int train2 = forward.trainIdx;

    //    if (query1 == train2) {
    //        float distance = forward.distance;
    //        matches.push_back(cv::DMatch(query2, query1, distance));
    //    }
    //}
}
