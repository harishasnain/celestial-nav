#include "star_detection.h"

std::vector<Star> StarDetection::detectStars(const cv::Mat &image) {
    cv::Mat preprocessed = preprocessImage(image);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(preprocessed, circles, cv::HOUGH_GRADIENT, 1, 30, 100, 30, 3, 10);
    
    std::vector<Star> stars;
    for (const auto &circle : circles) {
        Star star;
        star.position = cv::Point2f(circle[0], circle[1]);
        star.magnitude = circle[2]; // Use radius as a proxy for magnitude
        stars.push_back(star);
    }
    
    return stars;
}

cv::Mat StarDetection::preprocessImage(const cv::Mat &image) {
    cv::Mat gray, blurred;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    return blurred;
}
