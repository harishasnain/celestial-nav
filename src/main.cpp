// main.cpp
#include "user_interface.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

const double PI = 3.14159265358979323846;

ReferenceStarData parseStarLine(const std::string& line) {
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (iss >> token) {
        tokens.push_back(token);
    }

    if (tokens.size() < 8) {
        throw std::runtime_error("Invalid star data format");
    }

    double ra_hours, ra_minutes, ra_seconds;
    double dec_degrees, dec_minutes, dec_seconds;

    // Parse RA (Right Ascension)
    std::istringstream(tokens[6].substr(0, 2)) >> ra_hours;
    std::istringstream(tokens[6].substr(2, 2)) >> ra_minutes;
    std::istringstream(tokens[6].substr(4)) >> ra_seconds;

    // Parse Dec (Declination)
    std::istringstream(tokens[7].substr(0, 3)) >> dec_degrees;
    std::istringstream(tokens[7].substr(3, 2)) >> dec_minutes;
    std::istringstream(tokens[7].substr(5)) >> dec_seconds;

    double ra = (ra_hours + ra_minutes / 60 + ra_seconds / 3600) * 15 * PI / 180;
    double dec = (std::abs(dec_degrees) + dec_minutes / 60 + dec_seconds / 3600) * PI / 180;
    if (dec_degrees < 0) dec = -dec;

    double magnitude;
    std::istringstream(tokens[4]) >> magnitude;

    return {Eigen::Vector2d(ra, dec), magnitude};
}

int main(int argc, char* argv[]) {
    std::string testImagePath = "/home/haris/celestial-nav/test/test_images/test_image1.png";
    if (argc > 1) {
        testImagePath = argv[1];
    }

    try {
        ImageAcquisition imageAcq;
        std::vector<ReferenceStarData> referenceStars;

        // Load reference stars from catalog
        std::ifstream catalogFile("/home/haris/celestial-nav/data/star_catalog.dat");
        if (catalogFile.is_open()) {
            std::string line;
            while (std::getline(catalogFile, line)) {
                try {
                    referenceStars.push_back(parseStarLine(line));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing line: " << e.what() << std::endl;
                    std::cerr << "Problematic line: " << line << std::endl;
                }
            }
            catalogFile.close();
        } else {
            throw std::runtime_error("Unable to open star catalog file");
        }

        if (referenceStars.empty()) {
            throw std::runtime_error("No reference stars loaded from catalog");
        }

        StarMatching starMatch(referenceStars);
        UserInterface ui(imageAcq, starMatch);
        ui.run(testImagePath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

