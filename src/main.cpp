// main.cpp
#include "user_interface.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string testImagePath;
    if (argc > 1) {
        testImagePath = argv[1];
    }

    try {
        ImageAcquisition imageAcq;
        // Initialize StarMatching with reference stars (you'll need to load these from your catalog)
        std::vector<ReferenceStarData> referenceStars;
        // Load reference stars from catalog here
        StarMatching starMatch(referenceStars);

        UserInterface ui(imageAcq, starMatch);
        ui.run(testImagePath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

