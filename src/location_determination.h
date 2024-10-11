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
    static std::vector<std::pair<Star, ReferenceStarData>> removeOutliers(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, const CameraParameters &cameraParams, const Eigen::Vector2d &initialGuess);
    static double calculateMedian(std::vector<double> values);
    static double calculateMAD(const std::vector<double>& values, double median);
    static double calculateGuessScore(const Eigen::Vector2d& guess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars);
    static Eigen::Vector2d nelderMeadOptimization(const Eigen::Vector2d& initialGuess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars);
    static Eigen::Vector2d multiStageOptimization(const Eigen::Vector2d& initialGuess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars);
    static Eigen::Vector2d calculateGradient(const Eigen::Vector2d& guess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars);
};