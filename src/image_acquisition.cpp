// image_acquisition.cpp

#include "image_acquisition.h"
#include <stdexcept>

ImageAcquisition::ImageAcquisition() {
    // Empty constructor
}

ImageAcquisition::~ImageAcquisition() {
    // Empty destructor
}

cv::Mat ImageAcquisition::captureImage(const std::string &testImagePath) {
    return loadTestImage(testImagePath);
}

cv::Mat ImageAcquisition::loadTestImage(const std::string &path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load test image: " + path);
    }
    return image;
}

