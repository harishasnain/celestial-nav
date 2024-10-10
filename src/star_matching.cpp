#include "star_matching.h"

StarMatching::StarMatching(const std::vector<ReferenceStarData> &referenceStars, double threshold)
    : referenceStars(referenceStars), matchingThreshold(threshold) {}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::matchStars(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = geometricVoting(detectedStars);
    
    std::vector<std::pair<Star, ReferenceStarData>> matches;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::MatrixXd::Index maxCol;
        double maxVote = votedMap.row(i).maxCoeff(&maxCol);
        if (maxVote > matchingThreshold) {
            matches.emplace_back(detectedStars[i], referenceStars[maxCol]);
        }
    }
    
    // Sort matches by vote value in descending order
    std::sort(matches.begin(), matches.end(), [&votedMap](const auto &a, const auto &b) {
        return votedMap(a.first.id, a.second.id) > votedMap(b.first.id, b.second.id);
    });

    // Keep only the top N matches
    const size_t maxMatches = 100; // Adjust this value as needed
    if (matches.size() > maxMatches) {
        matches.resize(maxMatches);
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
            
            // Calculate angular distance instead of Euclidean distance
            double angularDistance = std::acos(detectedRelative.normalized().dot(referenceRelative.normalized()));
            maxDistance = std::max(maxDistance, angularDistance);
            
            // Use a smaller sigma value for more selective matching
            double sigma = 0.01;
            votedMap(i, j) = std::exp(-angularDistance * angularDistance / (2 * sigma * sigma));
        }
    }
    
    return votedMap;
}

void StarMatching::setMatchingThreshold(double threshold) {
    matchingThreshold = threshold;
}

void StarMatching::setMaxMatches(size_t max) {
    maxMatches = max;
}

