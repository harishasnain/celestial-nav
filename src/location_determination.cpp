#ifndef PI
#define PI 3.14159265358979323846
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "location_determination.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <vector>
#include <cmath>

#include "central_log.cpp"
#include "log_limiter.h"

struct LocationFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;

    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars;
    const CameraParameters& cameraParams;
    double lst;

    LocationFunctor(const std::vector<std::pair<Star, ReferenceStarData>>& matched, const CameraParameters& params, double localSiderealTime)
        : matchedStars(matched), cameraParams(params), lst(localSiderealTime) {}

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
        double lat = x(0);
        double lon = x(1);

        for (size_t i = 0; i < matchedStars.size(); ++i) {
            const auto& match = matchedStars[i];
            Eigen::Vector2d skyCoords = LocationDetermination::imageToSkyCoordinates(match.first.position, cameraParams);
            double expectedRa, expectedDec;
            LocationDetermination::raDecToAltAz(match.second.position.x(), match.second.position.y(), lat, lon, lst, expectedRa, expectedDec);
            fvec(2*i) = skyCoords.x() - expectedRa;
            fvec(2*i+1) = skyCoords.y() - expectedDec;
        }
        return 0;
    }

    int inputs() const { return 2; }
    int values() const { return 2 * matchedStars.size(); }
};

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars, const std::chrono::system_clock::time_point& observationTime) {
    LOG_INFO("LocationDetermination::calculateInitialGuess - Starting initial guess calculation");
    LOG_INFO("LocationDetermination::calculateInitialGuess - Number of matched stars: " + std::to_string(matchedStars.size()));

    if (matchedStars.size() < 3) {
        LOG_ERROR("LocationDetermination::calculateInitialGuess - At least 3 matched stars are required for initial guess.");
        return Eigen::Vector2d(0, 0);
    }

    double lst = siderealTime(observationTime);
    LOG_DEBUG("LocationDetermination::calculateInitialGuess - Local Sidereal Time: " + std::to_string(lst) + " radians");

    const int gridSize = 30;
    const double minLat = 25.0 * PI / 180.0;
    const double maxLat = 47.0 * PI / 180.0;
    const double minLon = -83.0 * PI / 180.0;
    const double maxLon = -66.0 * PI / 180.0;

    LOG_DEBUG("LocationDetermination::calculateInitialGuess - Grid parameters: Size=" + std::to_string(gridSize) + 
              ", Lat range=[" + std::to_string(minLat * 180.0 / PI) + ", " + std::to_string(maxLat * 180.0 / PI) + "]" +
              ", Lon range=[" + std::to_string(minLon * 180.0 / PI) + ", " + std::to_string(maxLon * 180.0 / PI) + "]");

    const double latStep = (maxLat - minLat) / gridSize;
    const double lonStep = (maxLon - minLon) / gridSize;

    Eigen::Vector2d bestGuess(0, 0);
    double minError = std::numeric_limits<double>::max();

    LOG_INFO("LocationDetermination::calculateInitialGuess - Starting grid search");
    for (int i = 0; i <= gridSize; ++i) {
        for (int j = 0; j <= gridSize; ++j) {
            double lat = minLat + i * latStep;
            double lon = minLon + j * lonStep;
            Eigen::Vector2d guess(lat, lon);

            LOG_DEBUG("LocationDetermination::calculateInitialGuess - Evaluating grid point: Lat=" + std::to_string(lat * 180.0 / PI) + ", Lon=" + std::to_string(lon * 180.0 / PI));

            double error = calculateAngularError(matchedStars, lat, lon, lst);

            LOG_DEBUG("LocationDetermination::calculateInitialGuess - Grid point (" + std::to_string(i) + ", " + std::to_string(j) + "): Lat=" + 
                      std::to_string(lat * 180.0 / PI) + ", Lon=" + std::to_string(lon * 180.0 / PI) + 
                      ", Error=" + std::to_string(error));

            if (error < minError) {
                minError = error;
                bestGuess = guess;
                LOG_INFO("LocationDetermination::calculateInitialGuess - New best guess: Lat=" + std::to_string(bestGuess.x() * 180.0 / PI) + 
                         ", Lon=" + std::to_string(bestGuess.y() * 180.0 / PI) + 
                         ", Error=" + std::to_string(minError));
            }
        }
    }

    LOG_INFO("LocationDetermination::calculateInitialGuess - Final best guess: Lat=" + std::to_string(bestGuess.x() * 180.0 / PI) + 
             ", Lon=" + std::to_string(bestGuess.y() * 180.0 / PI) + 
             ", Error=" + std::to_string(minError));
    return bestGuess;
}

double LocationDetermination::calculateAngularError(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars, double lat, double lon, double lst) {
    LOG_DEBUG("LocationDetermination::calculateAngularError - Calculating angular error for Lat=" + std::to_string(lat * 180.0 / PI) + "°, Lon=" + std::to_string(lon * 180.0 / PI) + "°");
    double totalError = 0;
    
    const int maxStars = 10;
    int numPairs = std::min(static_cast<int>(matchedStars.size()), maxStars);

    LOG_DEBUG("LocationDetermination::calculateAngularError - Using " + std::to_string(numPairs) + " stars for angular error calculation");

    for (int i = 0; i < numPairs; ++i) {
        for (int j = i + 1; j < numPairs; ++j) {
            const auto& star1 = matchedStars[i];
            const auto& star2 = matchedStars[j];

            double observedAngle = calculateAngleBetweenStars(
                Eigen::Vector2d(star1.first.position.x, star1.first.position.y),
                Eigen::Vector2d(star2.first.position.x, star2.first.position.y)
            );
            double expectedAngle = calculateExpectedAngleBetweenStars(star1.second, star2.second, lat, lon, lst);

            double pairError = std::pow(observedAngle - expectedAngle, 2);
            totalError += pairError;

            LOG_DEBUG("LocationDetermination::calculateAngularError - Star pair (" + std::to_string(i) + ", " + std::to_string(j) + "): Observed angle=" + std::to_string(observedAngle) 
                + ", Expected angle=" + std::to_string(expectedAngle) + ", Pair error=" + std::to_string(pairError));
        }
    }

    LOG_DEBUG("LocationDetermination::calculateAngularError - Total angular error: " + std::to_string(totalError));
    return totalError;
}

double LocationDetermination::calculateAngleBetweenStars(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2) {
    double angle = std::acos(std::min(1.0, std::max(-1.0, star1.normalized().dot(star2.normalized()))));
    LOG_DEBUG("LocationDetermination::calculateAngleBetweenStars - Angle between stars: " + std::to_string(angle));
    return angle;
}

double LocationDetermination::calculateExpectedAngleBetweenStars(const ReferenceStarData& star1, const ReferenceStarData& star2, double lat, double lon, double lst) {
    double alt1, az1, alt2, az2;
    raDecToAltAz(star1.position.x(), star1.position.y(), lat, lon, lst, alt1, az1);
    raDecToAltAz(star2.position.x(), star2.position.y(), lat, lon, lst, alt2, az2);

    LOG_DEBUG("LocationDetermination::calculateExpectedAngleBetweenStars - Star 1: RA=" + std::to_string(star1.position.x()) + ", Dec=" + std::to_string(star1.position.y()) + 
              ", Alt=" + std::to_string(alt1) + ", Az=" + std::to_string(az1));
    LOG_DEBUG("LocationDetermination::calculateExpectedAngleBetweenStars - Star 2: RA=" + std::to_string(star2.position.x()) + ", Dec=" + std::to_string(star2.position.y()) + 
              ", Alt=" + std::to_string(alt2) + ", Az=" + std::to_string(az2));

    Eigen::Vector3d vec1(std::cos(alt1) * std::sin(az1), std::cos(alt1) * std::cos(az1), std::sin(alt1));
    Eigen::Vector3d vec2(std::cos(alt2) * std::sin(az2), std::cos(alt2) * std::cos(az2), std::sin(alt2));

    double angle = std::acos(vec1.dot(vec2));
    LOG_DEBUG("LocationDetermination::calculateExpectedAngleBetweenStars - Expected angle between stars: " + std::to_string(angle));
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
    auto duration = time.time_since_epoch();
    auto days = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<86400>>>(duration).count();
    double T = days / 36525.0;
    double theta = 280.46061837 + 360.98564736629 * days + 0.000387933 * T * T - T * T * T / 38710000.0;
    return std::fmod(theta, 360.0) * PI / 180.0;
}


Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, 
                                                         const std::chrono::system_clock::time_point &observationTime,
                                                         const CameraParameters &cameraParams,
                                                         const Eigen::Vector2d &initialGuess) {
    LOG_INFO("LocationDetermination::determineLocation - Starting location determination");
    LOG_INFO("LocationDetermination::determineLocation - Number of matched stars: " + std::to_string(matchedStars.size()));

    Eigen::Vector2d guess = initialGuess;
    if (guess.isZero()) {
        LOG_INFO("LocationDetermination::determineLocation - Initial guess is zero, calculating initial guess");
        guess = calculateInitialGuess(matchedStars, observationTime);
    }

    double lst = siderealTime(observationTime);
    LOG_DEBUG("LocationDetermination::determineLocation - Local Sidereal Time: " + std::to_string(lst) + " radians");

    LocationFunctor functor(matchedStars, cameraParams, lst);
    Eigen::NumericalDiff<LocationFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<LocationFunctor>> lm(numDiff);

    Eigen::VectorXd x(2);
    x << guess.x(), guess.y();
    LOG_DEBUG("LocationDetermination::determineLocation - Starting optimization with initial guess: Lat=" + std::to_string(guess.x() * 180.0 / PI) + ", Lon=" + std::to_string(guess.y() * 180.0 / PI));

    int status = lm.minimize(x);

    Eigen::Vector2d result(x(0), x(1));
    LOG_INFO("LocationDetermination::determineLocation - Optimization status: " + std::to_string(status));
    LOG_INFO("LocationDetermination::determineLocation - Final location estimate: Lat=" + std::to_string(result.x() * 180.0 / PI) + ", Lon=" + std::to_string(result.y() * 180.0 / PI));
    return result;
}

Eigen::Vector2d LocationDetermination::imageToSkyCoordinates(const cv::Point2f& imagePoint, const CameraParameters& cameraParams) {
    double x = (imagePoint.x - cameraParams.centerX) * cameraParams.pixelSize;
    double y = (imagePoint.y - cameraParams.centerY) * cameraParams.pixelSize;
    double f = cameraParams.focalLength * cameraParams.pixelSize;

    double ra = std::atan2(x, f);
    double dec = std::atan2(y * std::cos(ra), f);

    return Eigen::Vector2d(ra, dec);
}