#include "star_matching.h"
#include <cmath>
#include <algorithm>
#include <numeric>

constexpr double PI = 3.14159265358979323846;

StarMatching::StarMatching(const std::vector<ReferenceStarData> &referenceStars)
    : referenceStars(referenceStars) {
    auto accessor = [](const ReferenceStarData& a, int dim) -> double { return a.position[dim]; };
    kdtree = KDTree::KDTree<2, ReferenceStarData, std::function<double(const ReferenceStarData&, int)>>(accessor);
    for (const auto &star : referenceStars) {
        kdtree.insert(star);
    }
    kdtree.optimize();
}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::matchStars(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = geometricVoting(detectedStars);
    
    double adaptiveThreshold = calculateAdaptiveThreshold(votedMap);
    
    std::vector<std::pair<Star, ReferenceStarData>> matches;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::MatrixXd::Index maxCol;
        double maxVote = votedMap.row(i).maxCoeff(&maxCol);
        if (maxVote > adaptiveThreshold) {
            matches.emplace_back(detectedStars[i], referenceStars[maxCol]);
        }
    }
    
    // Sort matches by vote value in descending order
    std::sort(matches.begin(), matches.end(), [&votedMap, &detectedStars, this](const auto &a, const auto &b) {
        size_t aIndex = &a.first - &detectedStars[0];
        size_t bIndex = &b.first - &detectedStars[0];
        size_t aRefIndex = &a.second - &referenceStars[0];
        size_t bRefIndex = &b.second - &referenceStars[0];
        return votedMap(aIndex, aRefIndex) > votedMap(bIndex, bRefIndex);
    });

    // Keep only the top N matches
    if (matches.size() > maxMatches) {
        matches.resize(maxMatches);
    }
    
    // Reject outliers
    matches = rejectOutliers(matches);
    
    std::cout << "Number of matches before filtering: " << matches.size() << std::endl;
    std::cout << "Adaptive threshold: " << adaptiveThreshold << std::endl;
    std::cout << "Max vote value: " << votedMap.maxCoeff() << std::endl;
    
    return matches;
}

Eigen::MatrixXd StarMatching::geometricVoting(const std::vector<Star> &detectedStars) {
    Eigen::MatrixXd votedMap = Eigen::MatrixXd::Zero(detectedStars.size(), referenceStars.size());
    
    if (detectedStars.empty() || referenceStars.empty()) {
        return votedMap;
    }

    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::Vector2d detectedPos(detectedStars[i].position.x, detectedStars[i].position.y);
        
        for (size_t j = 0; j < referenceStars.size(); ++j) {
            Eigen::Vector2d referencePos = referenceStars[j].position;
            
            double angularDistance = std::acos(std::sin(detectedPos.y()) * std::sin(referencePos.y()) +
                                               std::cos(detectedPos.y()) * std::cos(referencePos.y()) *
                                               std::cos(detectedPos.x() - referencePos.x()));
            
            double sigma = 0.2; // Increased from 0.1 to allow for more matches
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

double StarMatching::calculateAdaptiveThreshold(const Eigen::MatrixXd &votedMap) {
    double sum = 0.0;
    double sq_sum = 0.0;
    int count = 0;

    for (int i = 0; i < votedMap.rows(); ++i) {
        for (int j = 0; j < votedMap.cols(); ++j) {
            double vote = votedMap(i, j);
            sum += vote;
            sq_sum += vote * vote;
            ++count;
        }
    }

    double mean = sum / count;
    double variance = (sq_sum / count) - (mean * mean);
    double stdev = std::sqrt(variance);

    return mean + 0.5 * stdev; // Changed from stdev to 0.5 * stdev
}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::rejectOutliers(const std::vector<std::pair<Star, ReferenceStarData>> &matches) {
    if (matches.size() < 4) return matches;  // Need at least 4 matches for meaningful statistics

    std::vector<double> distances;
    distances.reserve(matches.size());
    
    for (const auto &match : matches) {
        Eigen::Vector2d detectedPos(match.first.position.x, match.first.position.y);
        Eigen::Vector2d referencePos = match.second.position;
        distances.push_back((detectedPos - referencePos).norm());
    }
    
    std::sort(distances.begin(), distances.end());
    double q1 = distances[distances.size() / 4];
    double q3 = distances[3 * distances.size() / 4];
    double iqr = q3 - q1;
    double lower_bound = q1 - 1.5 * iqr;
    double upper_bound = q3 + 1.5 * iqr;
    
    std::vector<std::pair<Star, ReferenceStarData>> filteredMatches;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (distances[i] >= lower_bound && distances[i] <= upper_bound) {
            filteredMatches.push_back(matches[i]);
        }
    }
    
    return filteredMatches;
}