#include "star_matching.h"

StarMatching::StarMatching(const std::vector<ReferenceStarData> &referenceStars)
    : referenceStars(referenceStars) {}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::matchStars(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = geometricVoting(detectedStars);
    
    std::vector<std::pair<Star, ReferenceStarData>> matches;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::MatrixXd::Index maxCol;
        votedMap.row(i).maxCoeff(&maxCol);
        if (votedMap(i, maxCol) > 0.5) { // Adjust this threshold as needed
            matches.emplace_back(detectedStars[i], referenceStars[maxCol]);
        }
    }
    
    return matches;
}

Eigen::MatrixXd StarMatching::geometricVoting(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = Eigen::MatrixXd::Zero(detectedStars.size(), referenceStars.size());
    
    // Calculate the center of mass for detected stars
    Eigen::Vector2d detectedCenterOfMass = Eigen::Vector2d::Zero();
    for (const auto &star : detectedStars) {
        detectedCenterOfMass += Eigen::Vector2d(star.position.x, star.position.y);
    }
    detectedCenterOfMass /= detectedStars.size();

    // Calculate the center of mass for reference stars
    Eigen::Vector2d referenceCenterOfMass = Eigen::Vector2d::Zero();
    for (const auto &star : referenceStars) {
        referenceCenterOfMass += star.position;
    }
    referenceCenterOfMass /= referenceStars.size();

    double maxDistance = 0.0;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::Vector2d detectedPos(detectedStars[i].position.x, detectedStars[i].position.y);
        Eigen::Vector2d detectedRelative = detectedPos - detectedCenterOfMass;
        
        for (size_t j = 0; j < referenceStars.size(); ++j) {
            Eigen::Vector2d referenceRelative = referenceStars[j].position - referenceCenterOfMass;
            
            double distance = (detectedRelative - referenceRelative).norm();
            maxDistance = std::max(maxDistance, distance);
            
            votedMap(i, j) = std::exp(-distance * distance / (2 * 0.1 * 0.1)); // Gaussian voting
        }
    }
    
    return votedMap;
}

