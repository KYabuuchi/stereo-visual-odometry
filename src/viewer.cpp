#include "viewer.hpp"
#include "params.hpp"

namespace Viewer
{
void init()
{
    cv::namedWindow(Params::WINDOW_NAME, cv::WINDOW_NORMAL);
}

bool show(cv::Mat cur_left_image, cv::Mat cur_right_image, cv::Mat pre_left_image, cv::Mat pre_right_image)
{
    cv::Mat merge1, merge2;
    cv::hconcat(cur_left_image, cur_right_image, merge1);
    cv::hconcat(pre_left_image, pre_right_image, merge2);
    cv::vconcat(merge1, merge2, merge2);
    cv::imshow(Params::WINDOW_NAME, merge2);

    return true;
}

}  // namespace Viewer