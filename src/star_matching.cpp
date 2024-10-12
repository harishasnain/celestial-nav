#include "star_matching.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

constexpr double PI = 3.14159265358979323846;

// Add this at the beginning of the file
std::ofstream starMatchingLogFile("star_matching.log");

#define LOG(x) do { \
    std::stringstream ss; \
    ss << "[" << __func__ << "] " << x; \
    std::cout << ss.str() << std::endl; \
    starMatchingLogFile << ss.str() << std::endl; \
} while(0)

StarMatching::StarMatching(const std::vector<ReferenceStarData> &referenceStars)
    : referenceStars(referenceStars) {
    LOG("Initializing StarMatching with " << referenceStars.size() << " reference stars");
    auto accessor = [](const ReferenceStarData& a, int dim) -> double { return a.position[dim]; };
    kdtree = KDTree::KDTree<2, ReferenceStarData, std::function<double(const ReferenceStarData&, int)>>(accessor);
    for (const auto &star : referenceStars) {
        kdtree.insert(star);
    }
    kdtree.optimize();
    LOG("KD-tree built and optimized");
}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::matchStars(const std::vector<Star> &detectedStars) {
    LOG("Starting star matching process");
    LOG("Number of detected stars: " << detectedStars.size());
    LOG("Number of reference stars: " << referenceStars.size());

    Eigen::MatrixXd votedMap = geometricVoting(detectedStars);
    
    LOG("Voted map dimensions: " << votedMap.rows() << "x" << votedMap.cols());
    
    double adaptiveThreshold = calculateAdaptiveThreshold(votedMap);
    LOG("Adaptive threshold: " << adaptiveThreshold);
    
    std::vector<std::pair<Star, ReferenceStarData>> matches;
    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::MatrixXd::Index maxCol;
        double maxVote = votedMap.row(i).maxCoeff(&maxCol);
        if (maxVote > adaptiveThreshold && isMatchProbable(detectedStars[i], referenceStars[maxCol], 1.0)) {
            matches.emplace_back(detectedStars[i], referenceStars[maxCol]);
        }
    }
    
    LOG("Number of matches before sorting: " << matches.size());

    // Sort matches by vote value in descending order
    std::sort(matches.begin(), matches.end(), [&votedMap, &detectedStars, this](const auto &a, const auto &b) {
        size_t aIndex = std::distance(detectedStars.begin(), std::find_if(detectedStars.begin(), detectedStars.end(), 
                                  [&](const Star& s) { return s.id == a.first.id; }));
        size_t bIndex = std::distance(detectedStars.begin(), std::find_if(detectedStars.begin(), detectedStars.end(), 
                                  [&](const Star& s) { return s.id == b.first.id; }));
        size_t aRefIndex = std::distance(referenceStars.begin(), std::find(referenceStars.begin(), referenceStars.end(), a.second));
        size_t bRefIndex = std::distance(referenceStars.begin(), std::find(referenceStars.begin(), referenceStars.end(), b.second));
        return votedMap(aIndex, aRefIndex) > votedMap(bIndex, bRefIndex);
    });

    // Keep only the top 100 matches
    if (matches.size() > 100) {
        matches.resize(100);
        LOG("Filtered to top 100 matches");
    }
    
    LOG("Number of matches after filtering: " << matches.size());

    // Reject outliers
    matches = rejectOutliers(matches);
    
    LOG("Number of matches after rejecting outliers: " << matches.size());
    LOG("Max vote value: " << votedMap.maxCoeff());
    
    return matches;
}

Eigen::MatrixXd StarMatching::geometricVoting(const std::vector<Star> &detectedStars) {
    LOG("Starting geometric voting");
    Eigen::MatrixXd votedMap = Eigen::MatrixXd::Zero(detectedStars.size(), referenceStars.size());
    
    if (detectedStars.empty() || referenceStars.empty()) {
        LOG("Warning: No detected stars or reference stars");
        return votedMap;
    }

    for (size_t i = 0; i < detectedStars.size(); ++i) {
        Eigen::Vector2d detectedPos(detectedStars[i].position.x, detectedStars[i].position.y);
        
        for (size_t j = 0; j < referenceStars.size(); ++j) {
            Eigen::Vector2d referencePos = referenceStars[j].position;
            
            // Calculate angular distance using the haversine formula
            double dlon = referencePos.x() - detectedPos.x();
            double dlat = referencePos.y() - detectedPos.y();
            double a = std::sin(dlat/2) * std::sin(dlat/2) + std::cos(detectedPos.y()) * std::cos(referencePos.y()) * std::sin(dlon/2) * std::sin(dlon/2);
            double angularDistance = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
            
            double sigma = 0.1; // Decreased from 0.2 to make matching more stringent
            votedMap(i, j) = std::exp(-angularDistance * angularDistance / (2 * sigma * sigma));
        }
    }
    
    LOG("Geometric voting completed");
    return votedMap;
}

void StarMatching::setMatchingThreshold(double threshold) {
    LOG("Setting matching threshold to " << threshold);
    matchingThreshold = threshold;
}

void StarMatching::setMaxMatches(size_t max) {
    LOG("Setting max matches to " << max);
    maxMatches = max;
}

double StarMatching::calculateAdaptiveThreshold(const Eigen::MatrixXd &votedMap) {
    LOG("Calculating adaptive threshold");
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

    double threshold = mean + 0.5 * stdev;
    LOG("Calculated adaptive threshold: " << threshold);
    return threshold;
}

std::vector<std::pair<Star, ReferenceStarData>> StarMatching::rejectOutliers(const std::vector<std::pair<Star, ReferenceStarData>> &matches) {
    LOG("Rejecting outliers from " << matches.size() << " matches");
    if (matches.size() < 4) {
        LOG("Not enough matches to reject outliers, returning original matches");
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
    
    LOG("Outlier rejection complete. Remaining matches: " << filteredMatches.size());
    return filteredMatches;
}

bool StarMatching::isMatchProbable(const Star& detectedStar, const ReferenceStarData& referenceStar, double threshold) {
    // Compare magnitudes (assuming they are in the same scale)
    double magnitudeDiff = std::abs(detectedStar.magnitude - referenceStar.magnitude);
    
    // You might need to adjust this threshold based on your specific use case
    return magnitudeDiff < threshold;
}

// Add this at the end of the file
StarMatching::~StarMatching() {
    starMatchingLogFile.close();
}