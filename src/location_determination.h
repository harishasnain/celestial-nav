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
    static void raDecToAltAz(double ra, double dec, double lat, double lon, double lst, double& alt, double& az);
    static Eigen::Vector2d imageToSkyCoordinates(const cv::Point2f& imagePoint, const CameraParameters& cameraParams);
    ~LocationDetermination();

private:
    static Eigen::Vector2d calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars, const std::chrono::system_clock::time_point &observationTime);
    static double siderealTime(const std::chrono::system_clock::time_point &time);
    static double calculateAngularError(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars, double lat, double lon, double lst);
    static double calculateAngleBetweenStars(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2);
    static double calculateExpectedAngleBetweenStars(const ReferenceStarData& star1, const ReferenceStarData& star2, double lat, double lon, double lst);
};