#include "preprocessing.h"

cv::Mat Preprocessing::preprocess(const cv::Mat &input) {
    cv::Mat result = input.clone();
    result = removeHotPixels(result);
    result = enhanceContrast(result);
    return result;
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

