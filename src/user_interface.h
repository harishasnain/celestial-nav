#pragma once
#include "image_acquisition.h"
#include "preprocessing.h"
#include "star_detection.h"
#include "star_matching.h"
#include "location_determination.h"

class UserInterface {
public: 
    UserInterface(ImageAcquisition &imageAcq, StarMatching &starMatch);
    void run(const std::string &testImagePath = "/home/haris/celestial-nav/test/test_images/test_image1.png");

private:
    ImageAcquisition &imageAcquisition;
    StarMatching &starMatching;
};
