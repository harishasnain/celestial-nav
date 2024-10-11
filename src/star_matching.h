#pragma once
#include "star_detection.h"
#include <Eigen/Dense>
#include <vector>
#include <kdtree++/kdtree.hpp>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

struct Star;  // Forward declaration

struct ReferenceStarData {
    Eigen::Vector2d position;  // (ra, dec) in radians
    double magnitude;

    bool operator==(const ReferenceStarData& other) const {
        return position == other.position && magnitude == other.magnitude;
    }
};

class StarMatching {
public:
    StarMatching(const std::vector<ReferenceStarData> &referenceStars);
    ~StarMatching();
    std::vector<std::pair<Star, ReferenceStarData>> matchStars(const std::vector<Star> &detectedStars);
    void setMatchingThreshold(double threshold);
    void setMaxMatches(size_t max);

private:
    std::vector<ReferenceStarData> referenceStars;
    KDTree::KDTree<2, ReferenceStarData, std::function<double(const ReferenceStarData&, int)>> kdtree;
    Eigen::MatrixXd geometricVoting(const std::vector<Star> &detectedStars);
    double calculateAdaptiveThreshold(const Eigen::MatrixXd &votedMap);
    std::vector<std::pair<Star, ReferenceStarData>> rejectOutliers(const std::vector<std::pair<Star, ReferenceStarData>> &matches);
    double matchingThreshold;
    size_t maxMatches = 100;

};
