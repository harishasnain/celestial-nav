#include "star_detection.h"
#include <algorithm>
#include "central_log.cpp"  // Ensure this is included for logging
#include "log_limiter.h"

std::vector<Star> StarDetection::detectStars(const cv::Mat &image) {
    LOG_INFO("StarDetection::detectStars - Starting star detection with image dimensions: " + 
             std::to_string(image.rows) + "x" + std::to_string(image.cols));

    cv::Mat preprocessed = preprocessImage(image);
    LOG_DEBUG("StarDetection::detectStars - Image preprocessed. Dimensions: " + 
              std::to_string(preprocessed.rows) + "x" + std::to_string(preprocessed.cols));

    cv::Mat binary;
    double thresholdValue = cv::threshold(preprocessed, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    LOG_DEBUG("StarDetection::detectStars - Applied binary threshold with Otsu's method. Threshold value: " + 
              std::to_string(thresholdValue));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    LOG_DEBUG("StarDetection::detectStars - Found " + std::to_string(contours.size()) + " contours");

    std::vector<Star> stars;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > 5) {  // Minimum area threshold
            cv::Moments m = cv::moments(contours[i]);
            Star star;
            star.position = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
            star.magnitude = area;  // Use area as a proxy for magnitude
            star.id = i;  // Assign an ID to the star
            stars.push_back(star);
            LOG_DEBUG("StarDetection::detectStars - Detected star with ID " + std::to_string(star.id) + " at position (" +
                      std::to_string(star.position.x) + ", " + std::to_string(star.position.y) + 
                      ") with magnitude " + std::to_string(star.magnitude) + " and area " + std::to_string(area));
        }
    }

    // Sort stars by magnitude (area) in descending order
    std::sort(stars.begin(), stars.end(), [](const Star &a, const Star &b) {
        return a.magnitude > b.magnitude;
    });
    LOG_DEBUG("StarDetection::detectStars - Sorted stars by magnitude");

    // Keep only the top 100 stars
    if (stars.size() > 100) {
        stars.resize(100);
        LOG_INFO("StarDetection::detectStars - Filtered to top 100 stars");
    }

    LOG_INFO("StarDetection::detectStars - Star detection completed with " + std::to_string(stars.size()) + " stars detected");
    return stars;
}

cv::Mat StarDetection::preprocessImage(const cv::Mat &image) {
    LOG_INFO("StarDetection::preprocessImage - Starting image preprocessing");

    cv::Mat gray, enhanced;
    if (image.channels() > 1) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        LOG_DEBUG("StarDetection::preprocessImage - Converted image to grayscale");
    } else {
        gray = image.clone();
    }
    cv::equalizeHist(gray, enhanced);
    LOG_DEBUG("StarDetection::preprocessImage - Equalized histogram");

    cv::GaussianBlur(enhanced, enhanced, cv::Size(3, 3), 0);
    LOG_DEBUG("StarDetection::preprocessImage - Applied Gaussian blur");

    LOG_INFO("StarDetection::preprocessImage - Image preprocessing completed");
    return enhanced;
}
