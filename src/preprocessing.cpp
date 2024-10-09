#include "preprocessing.h"

cv::Mat Preprocessing::preprocess(const cv::Mat& image) {
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    
    cv::Mat equalized;
    cv::equalizeHist(grayscale, equalized);
    
    cv::Mat blurred;
    cv::GaussianBlur(equalized, blurred, cv::Size(5, 5), 0);
    
    return blurred;
}

cv::Mat Preprocessing::removeHotPixels(const cv::Mat &input) {
    cv::Mat result;
    cv::medianBlur(input, result, 3);
    return result;
}

cv::Mat Preprocessing::enhanceContrast(const cv::Mat &input) {
    cv::Mat result;
    cv::equalizeHist(input, result);
    return result;
}

