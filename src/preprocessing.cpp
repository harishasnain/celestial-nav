#include "preprocessing.h"

#include "central_log.cpp"
#include "log_limiter.h"

cv::Mat Preprocessing::preprocess(const cv::Mat& image) {
    LOG_INFO("Preprocessing::preprocess - Starting image preprocessing with dimensions: " + 
             std::to_string(image.rows) + "x" + std::to_string(image.cols));
    
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    LOG_DEBUG("Preprocessing::preprocess - Converted image to grayscale. Dimensions: " + 
              std::to_string(grayscale.rows) + "x" + std::to_string(grayscale.cols));
    
    cv::Mat equalized;
    cv::equalizeHist(grayscale, equalized);
    LOG_DEBUG("Preprocessing::preprocess - Equalized histogram. Dimensions: " + 
              std::to_string(equalized.rows) + "x" + std::to_string(equalized.cols));
    
    cv::Mat blurred;
    cv::GaussianBlur(equalized, blurred, cv::Size(5, 5), 0);
    LOG_DEBUG("Preprocessing::preprocess - Applied Gaussian blur. Dimensions: " + 
              std::to_string(blurred.rows) + "x" + std::to_string(blurred.cols));
    
    LOG_INFO("Preprocessing::preprocess - Image preprocessing completed. Final dimensions: " + 
             std::to_string(blurred.rows) + "x" + std::to_string(blurred.cols));
    return blurred;
}

cv::Mat Preprocessing::removeHotPixels(const cv::Mat &input) {
    LOG_INFO("Preprocessing::removeHotPixels - Removing hot pixels from image with dimensions: " + 
             std::to_string(input.rows) + "x" + std::to_string(input.cols));
    cv::Mat result;
    cv::medianBlur(input, result, 3);
    LOG_DEBUG("Preprocessing::removeHotPixels - Applied median blur to remove hot pixels. Dimensions: " + 
              std::to_string(result.rows) + "x" + std::to_string(result.cols));
    return result;
}

cv::Mat Preprocessing::enhanceContrast(const cv::Mat &input) {
    LOG_INFO("Preprocessing::enhanceContrast - Enhancing image contrast for image with dimensions: " + 
             std::to_string(input.rows) + "x" + std::to_string(input.cols));
    cv::Mat result;
    cv::equalizeHist(input, result);
    LOG_DEBUG("Preprocessing::enhanceContrast - Applied histogram equalization for contrast enhancement. Dimensions: " + 
              std::to_string(result.rows) + "x" + std::to_string(result.cols));
    return result;
}
