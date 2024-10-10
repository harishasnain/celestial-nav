#include "star_detection.h"

std::vector<Star> StarDetection::detectStars(const cv::Mat &image) {
    cv::Mat preprocessed = preprocessImage(image);
    
    cv::Mat binary;
    cv::threshold(preprocessed, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Star> stars;
    for (const auto &contour : contours) {
        if (cv::contourArea(contour) > 5) {  // Minimum area threshold
            cv::Moments m = cv::moments(contour);
            Star star;
            star.position = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
            star.magnitude = cv::contourArea(contour);  // Use area as a proxy for magnitude
            stars.push_back(star);
        }
    }

    return stars;
}

cv::Mat StarDetection::preprocessImage(const cv::Mat &image) {
    cv::Mat gray, enhanced;
    if (image.channels() > 1) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    cv::equalizeHist(gray, enhanced);
    cv::GaussianBlur(enhanced, enhanced, cv::Size(3, 3), 0);
    return enhanced;
}
