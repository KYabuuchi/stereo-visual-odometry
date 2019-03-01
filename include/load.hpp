#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>

class Loader
{
public:
    Loader(const std::string& path_to_data)
        : m_path(path_to_data), m_file_num(0)
    {
        std::ifstream ifs(m_path + "/info.txt");
        if (not ifs) {
            std::cout << "can not open " << m_path << "/info.txt" << std::endl;
            abort();
        }

        while (!ifs.eof()) {
            std::string file_name;
            std::getline(ifs, file_name);
            if (not file_name.empty())
                file_names.push_back(file_name);
        }
    }

    bool load(cv::Mat& src1, cv::Mat& src2)
    {
        bool tmp = peek(src1, src2);
        m_file_num++;
        return tmp;
    }

    bool peek(cv::Mat& src1, cv::Mat& src2)
    {
        if (m_file_num >= file_names.size())
            return false;

        cv::Mat src = cv::imread(m_path + "/" + file_names.at(m_file_num));
        if (src.empty()) {
            std::cout << "can not open " << file_names.at(m_file_num);
            return false;
        }

        src1 = src.colRange(0, src.cols / 2);
        src2 = src.colRange(src.cols / 2, src.cols);
        return true;
    }

    void reset() { m_file_num = 0; }

private:
    const std::string m_path;
    int m_file_num;

    std::vector<std::string> file_names;
};