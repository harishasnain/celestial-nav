#include "user_interface.h"
#include <iostream>

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
    Eigen::Vector2d location = LocationDetermination::determineLocation(matchedStars);
    std::cout << "Estimated User Location (RA, Dec): (" << location.x() << ", " << location.y() << ")" << std::endl;
}

