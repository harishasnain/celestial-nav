#include "location_determination.h"

constexpr double PI = 3.14159265358979323846;

Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars,
                                                         const std::chrono::system_clock::time_point &observationTime) {
    
    Eigen::Vector2d position = calculateInitialGuess(matchedStars);
    std::cout << "Initial guess: (" << position.x() << ", " << position.y() << ")" << std::endl;
    
    for (int iteration = 0; iteration < 20; ++iteration) {
        Eigen::MatrixXd J(matchedStars.size(), 2);
        Eigen::VectorXd residuals(matchedStars.size());

        for (size_t i = 0; i < matchedStars.size(); ++i) {
            const auto &star = matchedStars[i].second;
            J.row(i) = calculateJacobian(position, star, observationTime);
            double measuredAltitude = std::asin(matchedStars[i].first.position.y);
            double calculatedAltitude = calculateAltitude(position, star, observationTime);
            residuals(i) = measuredAltitude - calculatedAltitude;
            
            std::cout << "Star " << i << ": Measured altitude: " << measuredAltitude 
                      << ", Calculated altitude: " << calculatedAltitude 
                      << ", Residual: " << residuals(i) << std::endl;
        }

        std::cout << "Jacobian:\n" << J << std::endl;
        std::cout << "Residuals: " << residuals.transpose() << std::endl;
        
        Eigen::Matrix2d JTJ = J.transpose() * J;
        double determinant = JTJ.determinant();
        std::cout << "Iteration " << iteration << " - Determinant: " << determinant << std::endl;
        
        if (determinant < 1e-10) {
            std::cout << "Matrix is near-singular. Breaking." << std::endl;
            break;
        }

        if (JTJ.hasNaN() || residuals.hasNaN()) {
            std::cout << "NaN values detected in JTJ or residuals. Breaking." << std::endl;
            break;
        }

        Eigen::Vector2d delta = JTJ.inverse() * J.transpose() * residuals;
        position += delta;

        std::cout << "Position: (" << position.x() << ", " << position.y() << "), Delta: " << delta.norm() << std::endl;

        if (delta.norm() < 1e-8) {
            std::cout << "Converged. Breaking." << std::endl;
            break;
        }
    }

    return position;
}

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars) {
    if (matchedStars.empty()) {
        std::cerr << "Error: No matched stars available for initial guess." << std::endl;
        return Eigen::Vector2d(0, 0);  // Return a default value
    }

    Eigen::Vector2d totalPosition = Eigen::Vector2d::Zero();
    double totalWeight = 0.0;
    
    for (const auto &match : matchedStars) {
        double weight = 1.0 / (match.second.magnitude + 1.0);  // Brighter stars have more weight
        
        if (!std::isfinite(weight) || !std::isfinite(match.second.position.x()) || !std::isfinite(match.second.position.y())) {
            std::cerr << "Warning: Invalid star data detected. Magnitude: " << match.second.magnitude 
                      << ", Position: (" << match.second.position.x() << ", " << match.second.position.y() << ")" << std::endl;
            continue;  // Skip this star
        }
        
        totalPosition += match.second.position * weight;
        totalWeight += weight;
    }
    
    if (totalWeight == 0.0) {
        std::cerr << "Error: Total weight is zero. Unable to calculate initial guess." << std::endl;
        return Eigen::Vector2d(0, 0);  // Return a default value
    }
    
    Eigen::Vector2d initialGuess = totalPosition / totalWeight;
    
    double ra = initialGuess.x();
    double dec = initialGuess.y();
    initialGuess.x() = dec;  // Latitude
    initialGuess.y() = std::fmod(ra, 2 * PI);  // Longitude, normalized to [0, 2π]
    if (initialGuess.y() > PI) {
        initialGuess.y() -= 2 * PI;  // Convert to range [-π, π]
    }
    
    std::cout << "Initial guess: (" << initialGuess.x() << ", " << initialGuess.y() << ")" << std::endl;
    
    return initialGuess;
}

Eigen::RowVector2d LocationDetermination::calculateJacobian(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime) {
    const double lat = position.x();
    const double lon = position.y();
    const double dec = star.position.y();
    const double ha = siderealTime(observationTime) - lon - star.position.x();

    Eigen::RowVector2d J;
    J(0) = -std::cos(dec) * std::sin(ha);
    J(1) = std::sin(lat) * std::cos(dec) * std::cos(ha) - std::cos(lat) * std::sin(dec);

    return J;
}

double LocationDetermination::calculateAltitude(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime) {
    double lat = position.x();
    double lon = position.y();
    double dec = star.position.y();
    double ra = star.position.x();
    double st = siderealTime(observationTime);
    double ha = st - lon - ra;

    double sinLat = std::sin(lat);
    double cosLat = std::cos(lat);
    double sinDec = std::sin(dec);
    double cosDec = std::cos(dec);
    double cosHa = std::cos(ha);

    double sinAlt = sinLat * sinDec + cosLat * cosDec * cosHa;
    return std::asin(sinAlt);
}

double LocationDetermination::siderealTime(const std::chrono::system_clock::time_point &time) {
    // Convert to Julian Date
    auto duration = time.time_since_epoch();
    double jd = std::chrono::duration<double, std::ratio<86400>>(duration).count() + 2440587.5;

    // Calculate Greenwich Mean Sidereal Time
    double T = (jd - 2451545.0) / 36525.0;
    double gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 
                  0.000387933 * T * T - T * T * T / 38710000.0;

    // Normalize to 0-360 degrees
    return std::fmod(gmst, 360.0) * PI / 180.0;
}