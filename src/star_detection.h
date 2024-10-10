#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Star {
    cv::Point2f position;
    float magnitude;
    double magnitude;
    size_t id;
};

class StarDetection {
public:
    static std::vector<Star> detectStars(const cv::Mat &image);

private:
    static cv::Mat preprocessImage(const cv::Mat &image);
};
