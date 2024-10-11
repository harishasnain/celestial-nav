#pragma once
#include "star_matching.h"
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include "camera_parameters.h"

class LocationDetermination {
public:
    static Eigen::Vector2d determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, 
                                             const std::chrono::system_clock::time_point &observationTime,
                                             const CameraParameters &cameraParams);

private:
    static Eigen::Vector2d calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars);
    static Eigen::RowVector2d calculateJacobian(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime);
    static double calculateAltitude(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime);
    static double siderealTime(const std::chrono::system_clock::time_point &time);
    static double calculateMeasuredAltitude(const Star &star, const CameraParameters &cameraParams);
};