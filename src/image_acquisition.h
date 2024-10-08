// image_acquisition.h

#pragma once
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory> // Added for std::unique_ptr

class ImageAcquisition {
public:
    ImageAcquisition();
    ~ImageAcquisition();
    cv::Mat captureImage(const std::string &testImagePath = "");

private:
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraConfiguration> config;

    // Added: Store the connection as a unique_ptr
    std::unique_ptr<libcamera::Signal::Connection> requestCompletedConnection;

    cv::Mat captureFromCamera();
    cv::Mat loadTestImage(const std::string &path);
    
    // Added: New method to handle completed requests
    void requestComplete(libcamera::Request *request);
};

