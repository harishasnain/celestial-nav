// main.cpp
#include "user_interface.h"
#include "star_matching.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

#include "central_log.h"

const double PI = 3.14159265358979323846;

ReferenceStarData parseStarLine(const std::string& line) {
    LOG_DEBUG("Parsing star line: '" + line + "'");
    if (line.length() < 103) {
        LOG_ERROR("Invalid star data format");
        throw std::runtime_error("Invalid star data format");
    }

    auto safeStod = [](const std::string& str, double defaultValue) {
        if (str.empty() || str == " ") return defaultValue;
        try {
            return std::stod(str);
        } catch (const std::exception&) {
            return defaultValue;
        }
    };

    // Extract RA (Right Ascension)
    double ra_hours = safeStod(line.substr(75, 2), 0);
    double ra_minutes = safeStod(line.substr(77, 2), 0);
    double ra_seconds = safeStod(line.substr(79, 4), 0);

    // Extract Dec (Declination)
    bool dec_negative = line[83] == '-';
    double dec_degrees = safeStod(line.substr(84, 2), 0);
    double dec_minutes = safeStod(line.substr(86, 2), 0);
    double dec_seconds = safeStod(line.substr(88, 2), 0);

    // Extract magnitude
    double magnitude = safeStod(line.substr(102, 5), 99.9);

    // Convert RA to radians
    double ra = (ra_hours + ra_minutes / 60.0 + ra_seconds / 3600.0) * 15 * PI / 180;

    // Convert Dec to radians
    double dec = (dec_degrees + dec_minutes / 60.0 + dec_seconds / 3600.0) * PI / 180;
    if (dec_negative) dec = -dec;

    LOG_DEBUG("Parsed star data: RA=" + std::to_string(ra) + ", Dec=" + std::to_string(dec) + ", Magnitude=" + std::to_string(magnitude));
    return ReferenceStarData{Eigen::Vector2d(ra, dec), magnitude};
}

int main(int argc, char* argv[]) {
    try {
        LOG_INFO("Starting Celestial Navigation Device");
    } catch (const std::exception& e) {
        std::cerr << "Error initializing log: " << e.what() << std::endl;
        return 1;
    }
    
    std::string testImagePath = "/home/haris/celestial-nav/test/test_images/test_image1.png";
    if (argc > 1) {
        testImagePath = argv[1];
        LOG_INFO("Using custom test image path: " + testImagePath);
    }

    try {
        ImageAcquisition imageAcq;
        std::vector<ReferenceStarData> referenceStars;

        LOG_INFO("Loading reference stars from catalog");
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
                    LOG_WARNING("Error parsing line " + std::to_string(lineNumber) + ": " + e.what());
                }
            }
            catalogFile.close();
            LOG_INFO("Successfully parsed " + std::to_string(successfullyParsedStars) + " stars from the catalog");
        } else {
            LOG_ERROR("Unable to open star catalog file");
            throw std::runtime_error("Unable to open star catalog file");
        }

        if (referenceStars.empty()) {
            LOG_ERROR("No reference stars loaded from catalog");
            throw std::runtime_error("No reference stars loaded from catalog");
        }

        StarMatching starMatch(referenceStars);
        starMatch.setMaxMatches(100);
        UserInterface ui(imageAcq, starMatch);
        
        std::chrono::system_clock::time_point observationTime = ui.getObservationTime();
        LOG_INFO("Using observation time from UserInterface");

        ui.run(testImagePath);
    } catch (const std::exception& e) {
        LOG_ERROR("Error: " + std::string(e.what()));
        return 1;
    }

    LOG_INFO("Celestial Navigation Device execution completed");
    return 0;
}
