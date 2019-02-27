#pragma once
#include "map.hpp"
#include <array>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

//namespace
//{
//const int PL = 0;
//const int PR = 1;
//const int CL = 2;
//const int CR = 3;
//}  // namespace

class Viewer
{
public:
    Viewer()
        : m_window_name("VIEW"),
          m_thread(std::make_shared<std::thread>(&Viewer::drawLoop, this)),
          m_reset_requested(false),
          m_stop_requested(false),
          m_updated(false) {}

    ~Viewer() { stop(); }

    void update(
        const std::array<cv::Mat, 4>& images,
        const std::vector<MapPointPtr>& mappoints)
    {
        std::lock_guard lock(m_mutex);
        m_mappoints = mappoints;
        m_images = images;
        m_updated = true;
    }

    void reset() { m_reset_requested = true; }
    void stop() { m_stop_requested = true; }

private:
    void drawLoop()
    {
        cv::namedWindow(m_window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(m_window_name, 640, 480);

        while (1) {
            // 描画
            if (m_updated) {
                m_updated = false;

                cv::Mat show, merge1, merge2;
                cv::hconcat(m_images.at(PL), m_images.at(PR), merge1);
                cv::hconcat(m_images.at(CL), m_images.at(CR), merge2);
                cv::vconcat(merge1, merge2, show);

                const cv::Size size = m_images.at(CL).size();
                const cv::Point2f OFFSET_PL(0, 0);
                const cv::Point2f OFFSET_PR(size.width, 0);
                const cv::Point2f OFFSET_CL(0, size.height);
                const cv::Point2f OFFSET_CR(size.width, size.height);

                for (const MapPointPtr p : m_mappoints) {
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
                cv::imshow(m_window_name, show);
            }

            if (m_reset_requested) {
                // TODO:
                m_reset_requested = false;
            }

            if (m_stop_requested) {
                break;
            }

            cv::waitKey(10);
        }

        std::cout << "viewer shut down " << std::endl;
    }

    std::array<cv::Mat, 4> m_images;
    std::vector<MapPointPtr> m_mappoints;
    const std::string m_window_name;
    std::shared_ptr<std::thread> m_thread;
    std::mutex m_mutex;

    bool m_reset_requested;
    bool m_stop_requested;
    bool m_updated;
};