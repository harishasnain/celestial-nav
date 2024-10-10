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

    // Find the RA and Dec tokens
    int raIndex = -1, decIndex = -1;
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        if (tokens[i].size() >= 6 && tokens[i + 1].size() >= 6) {
            if (raIndex == -1) {
                raIndex = i;
            } else {
                decIndex = i;
                break;
            }
        }
    }

    if (raIndex == -1 || decIndex == -1) {
        throw std::runtime_error("Could not find RA and Dec in the data");
    }

    // Parse RA (Right Ascension)
    double ra_hours = std::stod(tokens[raIndex].substr(0, 2));
    double ra_minutes = std::stod(tokens[raIndex].substr(2, 2));
    double ra_seconds = std::stod(tokens[raIndex].substr(4));

    // Parse Dec (Declination)
    double dec_degrees = std::stod(tokens[decIndex].substr(0, 3));
    double dec_minutes = std::stod(tokens[decIndex].substr(3, 2));
    double dec_seconds = std::stod(tokens[decIndex].substr(5));

    double ra = (ra_hours + ra_minutes / 60 + ra_seconds / 3600) * 15 * PI / 180;
    double dec = (std::abs(dec_degrees) + dec_minutes / 60 + dec_seconds / 3600) * PI / 180;
    if (dec_degrees < 0) dec = -dec;

    // Find magnitude
    double magnitude = 0.0;
    for (const auto& token : tokens) {
        try {
            magnitude = std::stod(token);
            break;
        } catch (const std::invalid_argument&) {
            continue;
        }
    }

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
            int lineNumber = 0;
            while (std::getline(catalogFile, line)) {
                ++lineNumber;
                try {
                    referenceStars.push_back(parseStarLine(line));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing line " << lineNumber << ": " << e.what() << std::endl;
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