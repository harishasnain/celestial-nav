#include "star_matching.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

#include "central_log.cpp"

constexpr double PI = 3.14159265358979323846;

StarMatching::StarMatching(const std::vector<ReferenceStarData> &referenceStars)
    : referenceStars(referenceStars) {
    LOG_INFO("StarMatching::StarMatching - Initializing with " + std::to_string(referenceStars.size()) + " reference stars");
    auto accessor = [](const ReferenceStarData& a, int dim) -> double { return a.position[dim]; };
    kdtree = KDTree::KDTree<2, ReferenceStarData, std::function<double(const ReferenceStarData&, int)>>(accessor);
    for (const auto &star : referenceStars) {
        kdtree.insert(star);
    }
    kdtree.optimize();
    LOG_DEBUG("StarMatching::StarMatching - KD-tree built and optimized");
}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::matchStars(const std::vector<Star> &detectedStars) {
    LOG_INFO("StarMatching::matchStars - Starting star matching process");
    LOG_DEBUG("StarMatching::matchStars - Number of detected stars: " + std::to_string(detectedStars.size()));
    LOG_DEBUG("StarMatching::matchStars - Number of reference stars: " + std::to_string(referenceStars.size()));

    Eigen::MatrixXd votedMap = geometricVoting(detectedStars);
    LOG_DEBUG("StarMatching::matchStars - Voted map dimensions: " + std::to_string(votedMap.rows()) + "x" + std::to_string(votedMap.cols()));

    double adaptiveThreshold = calculateAdaptiveThreshold(votedMap);
    LOG_DEBUG("StarMatching::matchStars - Adaptive threshold: " + std::to_string(adaptiveThreshold));

    std::vector<std::pair<Star, ReferenceStarData>> matches;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::MatrixXd::Index maxCol;
        double maxVote = votedMap.row(i).maxCoeff(&maxCol);
        if (maxVote > adaptiveThreshold && isMatchProbable(detectedStars[i], referenceStars[maxCol], 1.0)) {
            matches.emplace_back(detectedStars[i], referenceStars[maxCol]);
            LOG_DEBUG("StarMatching::matchStars - Match found for detected star ID " + std::to_string(detectedStars[i].id) +
                      " with reference star ID " + std::to_string(referenceStars[maxCol].id) + 
                      ". Vote: " + std::to_string(maxVote));
        }
    }
    
    LOG_DEBUG("StarMatching::matchStars - Number of matches before sorting: " + std::to_string(matches.size()));

    std::sort(matches.begin(), matches.end(), [&votedMap, &detectedStars, this](const auto &a, const auto &b) {
        size_t aIndex = std::distance(detectedStars.begin(), std::find_if(detectedStars.begin(), detectedStars.end(), 
                                  [&](const Star& s) { return s.id == a.first.id; }));
        size_t bIndex = std::distance(detectedStars.begin(), std::find_if(detectedStars.begin(), detectedStars.end(), 
                                  [&](const Star& s) { return s.id == b.first.id; }));
        size_t aRefIndex = std::distance(referenceStars.begin(), std::find(referenceStars.begin(), referenceStars.end(), a.second));
        size_t bRefIndex = std::distance(referenceStars.begin(), std::find(referenceStars.begin(), referenceStars.end(), b.second));
        return votedMap(aIndex, aRefIndex) > votedMap(bIndex, bRefIndex);
    });

    if (matches.size() > 100) {
        matches.resize(100);
        LOG_INFO("StarMatching::matchStars - Filtered to top 100 matches");
    }
    
    LOG_DEBUG("StarMatching::matchStars - Number of matches after filtering: " + std::to_string(matches.size()));

    matches = rejectOutliers(matches);
    LOG_DEBUG("StarMatching::matchStars - Number of matches after rejecting outliers: " + std::to_string(matches.size()));
    LOG_DEBUG("StarMatching::matchStars - Max vote value: " + std::to_string(votedMap.maxCoeff()));
    
    LOG_INFO("StarMatching::matchStars - Star matching process completed");
    return matches;
}

Eigen::MatrixXd StarMatching::geometricVoting(const std::vector<Star> &detectedStars) {
    LOG_INFO("StarMatching::geometricVoting - Starting geometric voting");
    Eigen::MatrixXd votedMap = Eigen::MatrixXd::Zero(detectedStars.size(), referenceStars.size());
    
    if (detectedStars.empty() || referenceStars.empty()) {
        LOG_WARNING("StarMatching::geometricVoting - No detected stars or reference stars");
        return votedMap;
    }

    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::Vector2d detectedPos(detectedStars[i].position.x, detectedStars[i].position.y);
        
        for (size_t j = 0; j < referenceStars.size(); ++j) {
            Eigen::Vector2d referencePos = referenceStars[j].position;
            
            double dlon = referencePos.x() - detectedPos.x();
            double dlat = referencePos.y() - detectedPos.y();
            double a = std::sin(dlat/2) * std::sin(dlat/2) + std::cos(detectedPos.y()) * std::cos(referencePos.y()) * std::sin(dlon/2) * std::sin(dlon/2);
            double angularDistance = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
            
            double sigma = 0.1;
            votedMap(i, j) = std::exp(-angularDistance * angularDistance / (2 * sigma * sigma));
            LOG_DEBUG("StarMatching::geometricVoting - Vote for detected star ID " + std::to_string(detectedStars[i].id) +
                      " and reference star ID " + std::to_string(referenceStars[j].id) + 
                      ": " + std::to_string(votedMap(i, j)));
        }
    }
    
    LOG_INFO("StarMatching::geometricVoting - Geometric voting completed");
    return votedMap;
}

void StarMatching::setMatchingThreshold(double threshold) {
    LOG_INFO("StarMatching::setMatchingThreshold - Setting matching threshold to " + std::to_string(threshold));
    matchingThreshold = threshold;
}

void StarMatching::setMaxMatches(size_t max) {
    LOG_INFO("StarMatching::setMaxMatches - Setting max matches to " + std::to_string(max));
    maxMatches = max;
}

double StarMatching::calculateAdaptiveThreshold(const Eigen::MatrixXd &votedMap) {
    LOG_INFO("StarMatching::calculateAdaptiveThreshold - Calculating adaptive threshold");
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

    double threshold = mean + 0.35 * stdev;
    LOG_DEBUG("StarMatching::calculateAdaptiveThreshold - Calculated adaptive threshold: " + std::to_string(threshold));
    return threshold;
}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::rejectOutliers(const std::vector<std::pair<Star, ReferenceStarData>> &matches) {
    LOG_INFO("StarMatching::rejectOutliers - Rejecting outliers from " + std::to_string(matches.size()) + " matches");
    if (matches.size() < 4) {
        LOG_WARNING("StarMatching::rejectOutliers - Not enough matches to reject outliers, returning original matches");
        return matches;
    }

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
    
    LOG_INFO("StarMatching::rejectOutliers - Outlier rejection complete. Remaining matches: " + std::to_string(filteredMatches.size()));
    return filteredMatches;
}

bool StarMatching::isMatchProbable(const Star& detectedStar, const ReferenceStarData& referenceStar, double threshold) {
    double magnitudeDiff = std::abs(detectedStar.magnitude - referenceStar.magnitude);
    return magnitudeDiff < threshold;
}

StarMatching::~StarMatching() {
    starMatchingLogFile.close();
}