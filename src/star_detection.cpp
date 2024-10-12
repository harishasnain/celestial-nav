#include "star_detection.h"
#include <algorithm>

std::vector<Star> StarDetection::detectStars(const cv::Mat &image) {
    cv::Mat preprocessed = preprocessImage(image);
    
    cv::Mat binary;
    cv::threshold(preprocessed, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Star> stars;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours[i]) > 5) {  // Minimum area threshold
            cv::Moments m = cv::moments(contours[i]);
            Star star;
            star.position = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
            star.magnitude = cv::contourArea(contours[i]);  // Use area as a proxy for magnitude
            star.id = i;  // Assign an ID to the star
            stars.push_back(star);
        }
    }

    // Sort stars by magnitude (area) in descending order
    std::sort(stars.begin(), stars.end(), [](const Star &a, const Star &b) {
        return a.magnitude > b.magnitude;
    });

    // Keep only the top 100 stars
    if (stars.size() > 100) {
        stars.resize(100);
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
