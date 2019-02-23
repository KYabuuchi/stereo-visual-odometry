#include "util.hpp"
#include "params.hpp"

bool readImage(int file_num, cv::Mat& src1, cv::Mat& src2)
{
    if (file_num > Params::MAX_FILE_NUM)
        file_num = Params::MAX_FILE_NUM;
    std::string file_path = "../data/VGA10CM/ZED_image" + std::to_string(file_num) + ".png";

    cv::Mat src = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    if (src.empty())
        return false;

    src1 = src.colRange(0, src.cols / 2);
    src2 = src.colRange(src.cols / 2, src.cols);
    return true;
}


cv::Mat calcPose(
    const std::vector<cv::Point2f>& cur_left,
    const std::vector<cv::Point2f>& pre_left)
{
    cv::Mat T = (cv::Mat_<double>(4, 4) << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);

    cv::Mat t, R, E, mask;
    E = cv::findEssentialMat(cur_left, pre_left, Params::ZED_INTRINSIC, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, cur_left, pre_left, Params::ZED_INTRINSIC, R, t, mask);
    R.copyTo(T.colRange(0, 3).rowRange(0, 3));
    t.copyTo(T.rowRange(0, 3).col(3));

    return T;
}
