#ifndef PI
#define PI 3.14159265358979323846
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "location_determination.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

// Add this at the beginning of the file
std::ofstream logFile("location_determination.log");

#define LOG(x) do { \
    std::stringstream ss; \
    ss << "[" << __func__ << "] " << x; \
    std::cout << ss.str() << std::endl; \
    logFile << ss.str() << std::endl; \
} while(0)

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars, const std::chrono::system_clock::time_point& observationTime) {
    LOG("Starting initial guess calculation");
    LOG("Number of matched stars: " << matchedStars.size());

    if (matchedStars.size() < 3) {
        LOG("Error: At least 3 matched stars are required for initial guess.");
        return Eigen::Vector2d(0, 0);
    }

    double lst = siderealTime(observationTime);
    LOG("Local Sidereal Time: " << lst);

    // Define a grid of potential locations along the east coast of the US
    const int gridSize = 30;
    const double minLat = 25.0 * PI / 180.0;
    const double maxLat = 47.0 * PI / 180.0;
    const double minLon = -83.0 * PI / 180.0;
    const double maxLon = -66.0 * PI / 180.0;

    LOG("Grid size: " << gridSize);
    LOG("Latitude range: " << minLat * 180.0 / PI << " to " << maxLat * 180.0 / PI);
    LOG("Longitude range: " << minLon * 180.0 / PI << " to " << maxLon * 180.0 / PI);

    const double latStep = (maxLat - minLat) / gridSize;
    const double lonStep = (maxLon - minLon) / gridSize;

    Eigen::Vector2d bestGuess(0, 0);
    double minError = std::numeric_limits<double>::max();

    LOG("Starting grid search");
    for (int i = 0; i <= gridSize; ++i) {
        for (int j = 0; j <= gridSize; ++j) {
            double lat = minLat + i * latStep;
            double lon = minLon + j * lonStep;
            Eigen::Vector2d guess(lat, lon);

            double error = calculateAngularError(matchedStars, lat, lon, lst);

            LOG("Grid point (" << i << ", " << j << "): Lat=" << lat * 180.0 / PI 
                << ", Lon=" << lon * 180.0 / PI << ", Error=" << error);

            if (error < minError) {
                minError = error;
                bestGuess = guess;
                LOG("New best guess: Lat=" << bestGuess.x() * 180.0 / PI 
                    << ", Lon=" << bestGuess.y() * 180.0 / PI << ", Error=" << minError);
            }
        }
    }

    LOG("Final best guess: Lat=" << bestGuess.x() * 180.0 / PI 
        << ", Lon=" << bestGuess.y() * 180.0 / PI << ", Error=" << minError);
    return bestGuess;
}

double LocationDetermination::calculateAngularError(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars, double lat, double lon, double lst) {
    LOG("Calculating angular error for Lat=" << lat * 180.0 / PI << ", Lon=" << lon * 180.0 / PI);
    double totalError = 0;
    int numPairs = matchedStars.size();

    for (int i = 0; i < numPairs; ++i) {
        for (int j = i + 1; j < numPairs; ++j) {
            const auto& star1 = matchedStars[i];
            const auto& star2 = matchedStars[j];

            Eigen::Vector2d pos1(star1.first.position.x, star1.first.position.y);
            Eigen::Vector2d pos2(star2.first.position.x, star2.first.position.y);

            double observedAngle = calculateAngleBetweenStars(pos1, pos2);
            double expectedAngle = calculateExpectedAngleBetweenStars(star1.second, star2.second, lat, lon, lst);

            double pairError = std::pow(observedAngle - expectedAngle, 2);
            totalError += pairError;

            LOG("Star pair (" << i << ", " << j << "): Observed angle=" << observedAngle 
                << ", Expected angle=" << expectedAngle << ", Pair error=" << pairError);
        }
    }

    LOG("Total angular error: " << totalError);
    return totalError;
}

double LocationDetermination::calculateAngleBetweenStars(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2) {
    double angle = std::acos(std::min(1.0, std::max(-1.0, star1.normalized().dot(star2.normalized()))));
    LOG("Angle between stars: " << angle);
    return angle;
}

double LocationDetermination::calculateExpectedAngleBetweenStars(const ReferenceStarData& star1, const ReferenceStarData& star2, double lat, double lon, double lst) {
    double alt1, az1, alt2, az2;
    raDecToAltAz(star1.position.x(), star1.position.y(), lat, lon, lst, alt1, az1);
    raDecToAltAz(star2.position.x(), star2.position.y(), lat, lon, lst, alt2, az2);

    LOG("Star 1: RA=" << star1.position.x() << ", Dec=" << star1.position.y() 
        << ", Alt=" << alt1 << ", Az=" << az1);
    LOG("Star 2: RA=" << star2.position.x() << ", Dec=" << star2.position.y() 
        << ", Alt=" << alt2 << ", Az=" << az2);

    Eigen::Vector3d vec1(std::cos(alt1) * std::sin(az1), std::cos(alt1) * std::cos(az1), std::sin(alt1));
    Eigen::Vector3d vec2(std::cos(alt2) * std::sin(az2), std::cos(alt2) * std::cos(az2), std::sin(alt2));

    double angle = std::acos(vec1.dot(vec2));
    LOG("Expected angle between stars: " << angle);
    return angle;
}

void LocationDetermination::raDecToAltAz(double ra, double dec, double lat, double lon, double lst, double& alt, double& az) {
    double ha = lst - ra;
    double sinAlt = std::sin(dec) * std::sin(lat) + std::cos(dec) * std::cos(lat) * std::cos(ha);
    alt = std::asin(sinAlt);
    double cosAz = (std::sin(dec) - std::sin(lat) * sinAlt) / (std::cos(lat) * std::cos(alt));
    az = std::acos(std::max(-1.0, std::min(1.0, cosAz)));
    if (std::sin(ha) > 0) {
        az = 2 * PI - az;
    }
}

double LocationDetermination::siderealTime(const std::chrono::system_clock::time_point &time) {
    // This is a simplified calculation and may not be accurate for all cases
    auto duration = time.time_since_epoch();
    auto days = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<86400>>>(duration).count();
    double T = days / 36525.0;
    double theta = 280.46061837 + 360.98564736629 * days + 0.000387933 * T * T - T * T * T / 38710000.0;
    return std::fmod(theta, 360.0) * PI / 180.0;
}

// Add this at the end of the file
LocationDetermination::~LocationDetermination() {
    logFile.close();
}

Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, 
                                                         const std::chrono::system_clock::time_point &observationTime,
                                                         const CameraParameters &cameraParams,
                                                         const Eigen::Vector2d &initialGuess) {
    LOG("Starting location determination");
    LOG("Number of matched stars: " << matchedStars.size());

    Eigen::Vector2d guess = initialGuess;
    if (guess.isZero()) {
        guess = calculateInitialGuess(matchedStars, observationTime);
    }

    // TODO: Implement optimization algorithm to refine the initial guess
    // This could involve using a non-linear least squares method like Levenberg-Marquardt

    LOG("Final location estimate: Lat=" << guess.x() * 180.0 / PI << ", Lon=" << guess.y() * 180.0 / PI);
    return guess;
}
