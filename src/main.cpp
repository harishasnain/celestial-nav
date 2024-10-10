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
    double ra_hours = 0, ra_minutes = 0, ra_seconds = 0;
    if (!line.substr(75, 2).empty()) ra_hours = std::stod(line.substr(75, 2));
    if (!line.substr(77, 2).empty()) ra_minutes = std::stod(line.substr(77, 2));
    if (!line.substr(79, 4).empty()) ra_seconds = std::stod(line.substr(79, 4));

    // Extract Dec (Declination)
    double dec_degrees = 0, dec_minutes = 0, dec_seconds = 0;
    bool dec_negative = line[83] == '-';
    if (!line.substr(84, 2).empty()) dec_degrees = std::stod(line.substr(84, 2));
    if (!line.substr(86, 2).empty()) dec_minutes = std::stod(line.substr(86, 2));
    if (!line.substr(88, 2).empty()) dec_seconds = std::stod(line.substr(88, 2));

    // Extract magnitude
    double magnitude = 0;
    if (!line.substr(102, 5).empty()) {
        try {
            magnitude = std::stod(line.substr(102, 5));
        } catch (const std::invalid_argument&) {
            // If magnitude can't be parsed, set it to a default value
            magnitude = 99.9; // Use 99.9 as a sentinel value for unknown magnitude
        }
    }

    // Convert RA to radians
    double ra = (ra_hours + ra_minutes / 60.0 + ra_seconds / 3600.0) * 15 * PI / 180;

    // Convert Dec to radians
    double dec = (dec_degrees + dec_minutes / 60.0 + dec_seconds / 3600.0) * PI / 180;
    if (dec_negative) dec = -dec;

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
                    std::cerr << "Warning: Error parsing line " << lineNumber << ": " << e.what() << std::endl;
                    // Optionally, you can still print the problematic line for debugging
                    // std::cerr << "Problematic line: " << line << std::endl;
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