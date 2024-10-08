#pragma once
#include "star_matching.h"
#include <Eigen/Dense>

class LocationDetermination {
public:
    static Eigen::Vector2d determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars);
};

