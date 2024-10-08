// image_acquisition.cpp

#include "image_acquisition.h"
#include <stdexcept>

ImageAcquisition::ImageAcquisition() {
    cameraManager = std::make_unique<libcamera::CameraManager>();
    cameraManager->start();
    
    auto cameras = cameraManager->cameras();
    if (cameras.empty()) {
        throw std::runtime_error("No cameras available");
    }
    
    camera = cameras[0];
    camera->acquire();
    
    config = camera->generateConfiguration({libcamera::StreamRole::StillCapture});
    config->at(0).pixelFormat = libcamera::formats::RGB888;
    config->at(0).size = libcamera::Size(2592, 1944); // Full resolution of Pi Camera V2
    config->validate();
    camera->configure(config.get());

    // Added: Connect the requestCompleted signal
    requestCompletedConnection = std::make_unique<libcamera::Signal::Connection>(
        camera->requestCompleted.connect(this, &ImageAcquisition::requestComplete)
    );
}

ImageAcquisition::~ImageAcquisition() {
    // Added: Disconnect the signal
    requestCompletedConnection.reset();

    camera->release();
    cameraManager->stop();
}

// Added: New method to handle completed requests
void ImageAcquisition::requestComplete(libcamera::Request *request) {
    // Handle the completed request here
}

cv::Mat ImageAcquisition::captureImage(const std::string &testImagePath) {
    if (!testImagePath.empty()) {
        return loadTestImage(testImagePath);
    }
    return captureFromCamera();
}

cv::Mat ImageAcquisition::captureFromCamera() {
    libcamera::FrameBufferAllocator *allocator = new libcamera::FrameBufferAllocator(camera);
    allocator->allocate(config->at(0).stream());
    
    camera->start();
    std::unique_ptr<libcamera::Request> request = camera->createRequest();
    if (!request) {
        throw std::runtime_error("Failed to create request");
    }
    
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>> &buffers = allocator->buffers(config->at(0).stream());
    int ret = request->addBuffer(config->at(0).stream(), buffers[0].get());
    if (ret < 0) {
        throw std::runtime_error("Failed to add buffer to request");
    }
    
    camera->queueRequest(request.get());
    
    libcamera::Request *completed_request = nullptr;
    // Changed: Use the class member requestComplete method
    requestCompletedConnection = std::make_unique<libcamera::SignalConnection>(
        camera->requestCompleted.connect([&](libcamera::Request *request) {
            if (!completed_request)
                completed_request = request;
        })
    );
    
    while (!completed_request) {
        camera->processEvents();
    }
    
    const libcamera::FrameBuffer *buffer = completed_request->buffers().begin()->second;
    const libcamera::FrameBuffer::Plane &plane = buffer->planes()[0];
    
    cv::Mat image(config->at(0).size.height, config->at(0).size.width, CV_8UC3);
    memcpy(image.data, plane.data(), plane.length);
    
    camera->stop();
    delete allocator;
    
    return image;
}

cv::Mat ImageAcquisition::loadTestImage(const std::string &path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load test image: " + path);
    }
    return image;
}

