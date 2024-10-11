#include "user_interface.h"
#include <iostream>
#include "camera_parameters.h"
#include <ctime>
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
    // Hardcoded observation time (May 1, 2023, 22:00:00 UTC)
    std::tm tm = {};
    tm.tm_year = 2024 - 1900;  // Years since 1900
    tm.tm_mon = 1;             // Months since January (0-11)
    tm.tm_mday = 5;            // Day of the month (1-31)
    tm.tm_hour = 22;           // Hours since midnight (0-23)
    tm.tm_min = 44;             // Minutes after the hour (0-59)
    tm.tm_sec = 0;             // Seconds after the minute (0-60)
    auto observationTime = std::chrono::system_clock::from_time_t(std::mktime(&tm));

    CameraParameters cameraParams{
        5116.28,  // focal length in pixels (22mm / 0.0043mm)
        2592.0,   // center X (5184 / 2)
        1728.0,   // center Y (3456 / 2)
        0.0043    // pixel size in mm
    };
    Eigen::Vector2d userLocation = LocationDetermination::determineLocation(matchedStars, observationTime, cameraParams, Eigen::Vector2d(0, 0));
    std::cout << "Estimated User Location (Lat, Lon): (" << userLocation.x() * 180.0 / PI << ", " << userLocation.y() * 180.0 / PI << ")" << std::endl;
}