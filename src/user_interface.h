#pragma once
#include "image_acquisition.h"
#include "preprocessing.h"
#include "star_detection.h"
#include "star_matching.h"
#include "location_determination.h"
#include <chrono>

class UserInterface {
public: 
    UserInterface(ImageAcquisition &imageAcq, StarMatching &starMatch);
    void run(const std::string &testImagePath = "/home/haris/celestial-nav/test/test_images/test_image1.png");
    std::chrono::system_clock::time_point getObservationTime() const;

private:
    ImageAcquisition &imageAcquisition;
    StarMatching &starMatching;
    std::chrono::system_clock::time_point observationTime;
};
