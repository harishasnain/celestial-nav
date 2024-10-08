#include "location_determination.h"

constexpr double PI = 3.14159265358979323846;

Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars,
                                                         const std::chrono::system_clock::time_point &observationTime) {
    Eigen::Vector2d position = calculateInitialGuess(matchedStars);
    
    for (int iteration = 0; iteration < 10; ++iteration) {
        Eigen::MatrixXd J(matchedStars.size(), 2);
        Eigen::VectorXd residuals(matchedStars.size());

        for (size_t i = 0; i < matchedStars.size(); ++i) {
            const auto &star = matchedStars[i].second;
            J.row(i) = calculateJacobian(position, star);
            double measuredAltitude = std::asin(matchedStars[i].first.position.y);
            double calculatedAltitude = calculateAltitude(position, star);
            residuals(i) = measuredAltitude - calculatedAltitude;
        }

        Eigen::Vector2d delta = (J.transpose() * J).inverse() * J.transpose() * residuals;
        position += delta;

        if (delta.norm() < 1e-8) {
            break;
        }
    }

    return position;
}

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars) {
    Eigen::Vector2d totalPosition = Eigen::Vector2d::Zero();
    for (const auto &match : matchedStars) {
        totalPosition += match.second.position;
    }
    return totalPosition / matchedStars.size();
}

Eigen::Matrix2d LocationDetermination::calculateJacobian(const Eigen::Vector2d &position, const ReferenceStarData &star) {
    const double lat = position.x();
    const double lon = position.y();
    const double dec = star.position.y();
    const double ha = siderealTime(std::chrono::system_clock::now()) - lon - star.position.x();

    Eigen::Matrix2d J;
    J(0, 0) = -std::cos(dec) * std::sin(ha);
    J(0, 1) = std::sin(lat) * std::cos(dec) * std::cos(ha) - std::cos(lat) * std::sin(dec);
    J(1, 0) = std::cos(lat) * std::sin(dec) - std::sin(lat) * std::cos(dec) * std::cos(ha);
    J(1, 1) = -std::cos(dec) * std::sin(ha);

    return J;
}

double LocationDetermination::calculateAltitude(const Eigen::Vector2d &position, const ReferenceStarData &star) {
    const double lat = position.x();
    const double lon = position.y();
    const double dec = star.position.y();
    const double ha = siderealTime(std::chrono::system_clock::now()) - lon - star.position.x();

    return std::asin(std::sin(lat) * std::sin(dec) + std::cos(lat) * std::cos(dec) * std::cos(ha));
}

double LocationDetermination::siderealTime(const std::chrono::system_clock::time_point &time) {
    // This is a simplified calculation of sidereal time
    auto duration = time.time_since_epoch();
    auto days = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<86400>>>(duration).count();
    return std::fmod(100.46 + 0.985647 * days, 360.0) * PI / 180.0;
}