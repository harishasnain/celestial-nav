// image_acquisition.h

#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ImageAcquisition {
public:
    ImageAcquisition();
    ~ImageAcquisition();
    cv::Mat captureImage(const std::string &testImagePath);

private:
    cv::Mat loadTestImage(const std::string &path);
};

