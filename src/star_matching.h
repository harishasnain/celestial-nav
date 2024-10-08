#pragma once
#include "star_detection.h"
#include <Eigen/Dense>
#include <vector>

struct ReferenceStarData {
    Eigen::Vector2d position; // RA and Dec
    double magnitude;
};

class StarMatching {
public:
    StarMatching(const std::vector<ReferenceStarData> &referenceStars);
    std::vector<std::pair<Star, ReferenceStarData>> matchStars(const std::vector<Star> &detectedStars);

private:
    std::vector<ReferenceStarData> referenceStars;
    Eigen::MatrixXd geometricVoting(const std::vector<Star> &detectedStars);
};
