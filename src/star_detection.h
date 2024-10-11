#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "star_matching.h"

class StarDetection {
public:
    static std::vector<Star> detectStars(const cv::Mat &image);

private:
    static cv::Mat preprocessImage(const cv::Mat &image);
};
