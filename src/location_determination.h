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
                                             const CameraParameters &cameraParams,
                                             const Eigen::Vector2d &initialGuess);

private:
    static Eigen::Vector2d calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, const std::chrono::system_clock::time_point &observationTime);
    static Eigen::Matrix2d calculateJacobian(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime);
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
    static void raDecToAltAz(double ra, double dec, double lat, double lon, double lst, double& alt, double& az);
    static double calculateAzimuth(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime);
    static double angularDistance(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2);
    static double estimateLatitude(double angle12, double angle23, double angle31);
    static double estimateLongitude(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2, const Eigen::Vector2d& star3, double lat, double lst);
    static double calculateHourAngle(const Eigen::Vector2d& star, double lat);
};