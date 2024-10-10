#pragma once
#include "star_detection.h"
#include <Eigen/Dense>
#include <vector>
#include <kdtree++/kdtree.hpp>
#include <algorithm>
#include <numeric>

struct ReferenceStarData {
    Eigen::Vector2d position; // RA and Dec
    double magnitude;
};

class StarMatching {
public:
    StarMatching(const std::vector<ReferenceStarData> &referenceStars);
    std::vector<std::pair<Star, ReferenceStarData>> matchStars(const std::vector<Star> &detectedStars);
    void setMatchingThreshold(double threshold);
    void setMaxMatches(size_t max);

private:
    std::vector<ReferenceStarData> referenceStars;
    KDTree::KDTree<2, ReferenceStarData> kdtree;
    Eigen::MatrixXd geometricVoting(const std::vector<Star> &detectedStars);
    double calculateAdaptiveThreshold(const Eigen::MatrixXd &votedMap);
    std::vector<std::pair<Star, ReferenceStarData>> rejectOutliers(const std::vector<std::pair<Star, ReferenceStarData>> &matches);
    double matchingThreshold;
    size_t maxMatches = 100;
};
