#include "user_interface.h"
#include <iostream>
#include "camera_parameters.h"
constexpr double PI = 3.14159265358979323846;

UserInterface::UserInterface(ImageAcquisition &imageAcq, StarMatching &starMatch)
    : imageAcquisition(imageAcq), starMatching(starMatch) {}

void UserInterface::run(const std::string &testImagePath) {
    std::cout << "Welcome to the Celestial Navigation Device!" << std::endl;
    
    std::cout << "Capturing image for star detection..." << std::endl;
    cv::Mat image = imageAcquisition.captureImage(testImagePath);
    
    std::cout << "Preprocessing image..." << std::endl;
    cv::Mat preprocessed = Preprocessing::preprocess(image);
    
    std::cout << "Detecting stars..." << std::endl;
    std::vector<Star> detectedStars = StarDetection::detectStars(preprocessed);
    std::cout << "Detected " << detectedStars.size() << " stars." << std::endl;
    
    std::cout << "Matching stars..." << std::endl;
    auto matchedStars = starMatching.matchStars(detectedStars);
    std::cout << "Matched " << matchedStars.size() << " stars." << std::endl;
    
    std::cout << "Determining location..." << std::endl;
    auto observationTime = std::chrono::system_clock::now();
    CameraParameters cameraParams{
        5116.28,  // focal length in pixels (22mm / 0.0043mm)
        2592.0,  // center X (5184 / 2)
        1728.0,  // center Y (3456 / 2)
        0.0043   // pixel size in mm
    };
    Eigen::Vector2d location = LocationDetermination::determineLocation(matchedStars, observationTime, cameraParams);
    std::cout << "Estimated User Location (Lat, Lon): (" << location.x() * 180.0 / PI << ", " << location.y() * 180.0 / PI << ")" << std::endl;
}

