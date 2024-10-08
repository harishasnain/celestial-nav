#pragma once
#include <opencv2/opencv.hpp>

class Preprocessing {
public:
    static cv::Mat preprocess(const cv::Mat &input);

private:
    static cv::Mat removeHotPixels(const cv::Mat &input);
    static cv::Mat enhanceContrast(const cv::Mat &input);
};
