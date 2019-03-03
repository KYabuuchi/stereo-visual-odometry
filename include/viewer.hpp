#pragma once
#include "map.hpp"
#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

class Viewer
{
public:
    Viewer();

    ~Viewer();

    void update(
        const std::array<cv::Mat, 4>& images,
        const cv::Mat Tcw,
        const std::vector<MapPointPtr>& mappoints);

    void reset();
    void stop();
    int waitKeyEver();
    int waitKeyOnce();

private:
    void drawLoop();

    std::array<cv::Mat, 4> m_images;
    std::vector<MapPointPtr> m_mappoints;

    const std::string m_window_name;

    std::shared_ptr<std::thread> m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    bool m_reset_requested;
    bool m_stop_requested;
    bool m_update_called;

    int m_last_key;
    cv::Mat1f m_Tcw;
};