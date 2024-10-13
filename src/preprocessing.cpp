#include "preprocessing.h"

#include "central_log.cpp"

cv::Mat Preprocessing::preprocess(const cv::Mat& image) {
    LOG_INFO("Starting image preprocessing");
    
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    LOG_DEBUG("Converted image to grayscale");
    
    cv::Mat equalized;
    cv::equalizeHist(grayscale, equalized);
    LOG_DEBUG("Equalized histogram");
    
    cv::Mat blurred;
    cv::GaussianBlur(equalized, blurred, cv::Size(5, 5), 0);
    LOG_DEBUG("Applied Gaussian blur");
    
    LOG_INFO("Image preprocessing completed");
    return blurred;
}

cv::Mat Preprocessing::removeHotPixels(const cv::Mat &input) {
    LOG_INFO("Removing hot pixels");
    cv::Mat result;
    cv::medianBlur(input, result, 3);
    LOG_DEBUG("Applied median blur to remove hot pixels");
    return result;
}

cv::Mat Preprocessing::enhanceContrast(const cv::Mat &input) {
    LOG_INFO("Enhancing image contrast");
    cv::Mat result;
    cv::equalizeHist(input, result);
    LOG_DEBUG("Applied histogram equalization for contrast enhancement");
    return result;
}

