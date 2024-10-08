#pragma once
#include "star_matching.h"
#include <Eigen/Dense>
#include <cmath>
#include <chrono>

class LocationDetermination {
public:
    static Eigen::Vector2d determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, 
                                             const std::chrono::system_clock::time_point &observationTime);

private:
    static Eigen::Vector2d calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars);
    static Eigen::Matrix2d calculateJacobian(const Eigen::Vector2d &position, const ReferenceStarData &star);
    static double calculateAltitude(const Eigen::Vector2d &position, const ReferenceStarData &star);
    static double siderealTime(const std::chrono::system_clock::time_point &time);
};