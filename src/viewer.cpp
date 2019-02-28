#include "viewer.hpp"
#include "params.hpp"
#include <chrono>

Viewer::Viewer()
    : m_window_name("VIEW"),
      m_thread(std::make_shared<std::thread>(&Viewer::drawLoop, this)),
      m_reset_requested(false),
      m_stop_requested(false),
      m_update_called(false),
      m_last_key(-1) {}

Viewer::~Viewer() { stop(); }

void Viewer::reset() { m_reset_requested = true; }
void Viewer::stop() { m_stop_requested = true; }

void Viewer::update(
    const std::array<cv::Mat, 4>& images,
    const std::vector<MapPointPtr>& mappoints)
{
    {
        std::lock_guard lock(m_mutex);
        m_mappoints.clear();
        for (const MapPointPtr point : mappoints) {
            m_mappoints.push_back(std::make_shared<MapPoint>(*point));
        }
        m_images = images;
        m_update_called = true;
    }
    for (cv::Mat& image : m_images) {
        if (image.empty())
            image = cv::Mat::zeros(Params::ZED_RESOLUTION, CV_8UC3);
    }
}


int Viewer::waitKeyEver()
{
    std::unique_lock lock(m_mutex);
    m_cv.wait(lock, [this] { return (m_last_key != -1); });
    return m_last_key;
}
void Viewer::drawLoop()
{
    cv::namedWindow(m_window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(m_window_name, 640 * 2, 480 * 2);

    while (1) {
        bool update = false;
        {
            std::lock_guard lock(m_mutex);
            update = m_update_called;
            m_update_called = false;
        }

        if (update) {
            cv::Mat show, merge1, merge2;
            cv::hconcat(m_images.at(PL), m_images.at(PR), merge1);
            cv::hconcat(m_images.at(CL), m_images.at(CR), merge2);
            cv::vconcat(merge1, merge2, show);

            const cv::Size2f size = m_images.at(CL).size();
            const cv::Point2f OFFSET_PL(0, 0);
            const cv::Point2f OFFSET_PR(size.width, 0);
            const cv::Point2f OFFSET_CL(0, size.height);
            const cv::Point2f OFFSET_CR(size.width, size.height);

            for (const MapPointPtr p : m_mappoints) {
                if (p->enable(PL))
                    cv::circle(show, p->preLeft() + OFFSET_PL, 1, CV_RGB(255, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(PR))
                    cv::circle(show, p->preRight() + OFFSET_PR, 1, CV_RGB(0, 255, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(CL))
                    cv::circle(show, p->curLeft() + OFFSET_CL, 1, CV_RGB(155, 0, 0), 0, cv::LineTypes::LINE_AA);
                if (p->enable(CR))
                    cv::circle(show, p->curRight() + OFFSET_CR, 1, CV_RGB(0, 155, 0), 0, cv::LineTypes::LINE_AA);

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

        // key event
        {
            std::lock_guard lock(m_mutex);
            m_last_key = cv::waitKey(5);
        }

        // wait for other thread action
        m_cv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "viewer shut down " << std::endl;
}