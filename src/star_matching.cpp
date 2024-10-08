#include "star_matching.h"

StarMatching::StarMatching(const std::vector<ReferenceStarData> &referenceStars)
    : referenceStars(referenceStars) {}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::matchStars(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = geometricVoting(detectedStars);
    
    std::vector<std::pair<Star, ReferenceStarData>> matches;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        for (size_t j = 0; j < referenceStars.size(); ++j) {
            if (votedMap(i, j) > votedMap.maxCoeff() / 2) {
                matches.emplace_back(detectedStars[i], referenceStars[j]);
            }
        }
    }
    
    return matches;
}

Eigen::MatrixXd StarMatching::geometricVoting(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = Eigen::MatrixXd::Zero(detectedStars.size(), referenceStars.size());
    
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        for (size_t j = 0; j < referenceStars.size(); ++j) {
            Eigen::Vector2d detectedPos(detectedStars[i].position.x, detectedStars[i].position.y);
            double distance = (detectedPos - referenceStars[j].position).norm();
            
            if (distance < 15) { // Threshold for voting
                votedMap(i, j) += 1;
            }
        }
    }
    
    return votedMap;
}

