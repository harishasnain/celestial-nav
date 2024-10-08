#include "location_determination.h"

Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars) {
    Eigen::Vector2d totalPosition = Eigen::Vector2d::Zero();
    
    for (const auto &match : matchedStars) {
        totalPosition += match.second.position;
    }
    
    return totalPosition / matchedStars.size();
}

