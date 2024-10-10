// main.cpp
#include "user_interface.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

const double PI = 3.14159265358979323846;

ReferenceStarData parseStarLine(const std::string& line) {
    if (line.length() < 103) {
        throw std::runtime_error("Invalid star data format");
    }

    // Extract RA (Right Ascension)
    int ra_hours = std::stoi(line.substr(75, 2));
    int ra_minutes = std::stoi(line.substr(77, 2));
    double ra_seconds = std::stod(line.substr(79, 4));

    // Extract Dec (Declination)
    int dec_degrees = std::stoi(line.substr(83, 3));
    int dec_minutes = std::stoi(line.substr(86, 2));
    int dec_seconds = std::stoi(line.substr(88, 2));

    // Extract magnitude
    double magnitude = std::stod(line.substr(102, 5));

    // Convert RA to radians
    double ra = (ra_hours + ra_minutes / 60.0 + ra_seconds / 3600.0) * 15 * PI / 180;

    // Convert Dec to radians
    double dec = (std::abs(dec_degrees) + dec_minutes / 60.0 + dec_seconds / 3600.0) * PI / 180;
    if (dec_degrees < 0) dec = -dec;

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
        std::ifstream catalogFile("/home/haris/celestial-nav/data/bsc5.dat");
        if (catalogFile.is_open()) {
            std::string line;
            int lineNumber = 0;
            int successfullyParsedStars = 0;
            while (std::getline(catalogFile, line)) {
                ++lineNumber;
                try {
                    referenceStars.push_back(parseStarLine(line));
                    ++successfullyParsedStars;
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing line " << lineNumber << ": " << e.what() << std::endl;
                    std::cerr << "Problematic line: " << line << std::endl;
                }
            }
            catalogFile.close();
            std::cout << "Successfully parsed " << successfullyParsedStars << " stars from the catalog." << std::endl;
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