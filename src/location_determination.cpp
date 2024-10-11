#include "location_determination.h"
#include "camera_parameters.h"

constexpr double PI = 3.14159265358979323846;

Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars,
                                                         const std::chrono::system_clock::time_point &observationTime,
                                                         const CameraParameters &cameraParams) {
    
    if (matchedStars.empty()) {
        std::cerr << "Error: No matched stars provided." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    Eigen::Vector2d position = calculateInitialGuess(matchedStars);
    std::cout << "Initial guess: (" << position.x() << ", " << position.y() << ")" << std::endl;
    
    if (std::isnan(position.x()) || std::isnan(position.y())) {
        std::cerr << "Error: Initial guess resulted in NaN values." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    // Ensure initial guess is within valid ranges
    position.x() = std::max(-PI/2, std::min(PI/2, position.x()));
    position.y() = std::fmod(position.y() + 3*PI, 2*PI) - PI;
    
    auto filteredStars = removeOutliers(matchedStars, cameraParams, position);
    std::cout << "Removed " << matchedStars.size() - filteredStars.size() << " outliers." << std::endl;

    const int MAX_ITERATIONS = 50;
    const double POSITION_TOLERANCE = 1e-8;
    const double RESIDUAL_TOLERANCE = 1e-6;
    double prev_ssr = std::numeric_limits<double>::max();

    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        Eigen::MatrixXd J(filteredStars.size(), 2);
        Eigen::VectorXd residuals(filteredStars.size());

        for (size_t i = 0; i < filteredStars.size(); ++i) {
            const auto &star = filteredStars[i].second;
            J.row(i) = calculateJacobian(position, star, observationTime);
            double measuredAltitude = calculateMeasuredAltitude(filteredStars[i].first, cameraParams);
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

        double ssr = residuals.squaredNorm();
        double relative_ssr_change = std::abs(ssr - prev_ssr) / prev_ssr;

        if (delta.norm() < POSITION_TOLERANCE || relative_ssr_change < RESIDUAL_TOLERANCE) {
            std::cout << "Converged. Breaking." << std::endl;
            break;
        }

        prev_ssr = ssr;

        if (iteration == MAX_ITERATIONS - 1) {
            std::cout << "Warning: Maximum iterations reached without convergence." << std::endl;
        }

        std::cout << "Iteration " << iteration << ":" << std::endl;
        std::cout << "  Position: (" << position.x() << ", " << position.y() << ")" << std::endl;
        std::cout << "  Sum of squared residuals: " << residuals.squaredNorm() << std::endl;
        std::cout << "  Determinant of JTJ: " << determinant << std::endl;
    }

    std::cout << "Final position: (" << position.x() << ", " << position.y() << ")" << std::endl;
    return position;
}

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars) {
    if (matchedStars.empty()) {
        std::cerr << "Error: No matched stars available for initial guess." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto &match : matchedStars) {
        Eigen::Vector3d star_vec(
            std::cos(match.second.position.y()) * std::cos(match.second.position.x()),
            std::cos(match.second.position.y()) * std::sin(match.second.position.x()),
            std::sin(match.second.position.y())
        );
        centroid += star_vec;
    }
    centroid.normalize();

    // Convert to geodetic coordinates
    double lat = std::atan2(centroid.z(), std::sqrt(centroid.x()*centroid.x() + centroid.y()*centroid.y()));
    double lon = std::atan2(centroid.y(), centroid.x());

    return Eigen::Vector2d(lat, lon);
}

Eigen::RowVector2d LocationDetermination::calculateJacobian(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime) {
    const double h = 1e-8; // Step size for numerical differentiation
    Eigen::RowVector2d J;

    // Partial derivative with respect to latitude
    Eigen::Vector2d pos_lat_plus = position + Eigen::Vector2d(h, 0);
    Eigen::Vector2d pos_lat_minus = position - Eigen::Vector2d(h, 0);
    J(0) = (calculateAltitude(pos_lat_plus, star, observationTime) - calculateAltitude(pos_lat_minus, star, observationTime)) / (2 * h);

    // Partial derivative with respect to longitude
    Eigen::Vector2d pos_lon_plus = position + Eigen::Vector2d(0, h);
    Eigen::Vector2d pos_lon_minus = position - Eigen::Vector2d(0, h);
    J(1) = (calculateAltitude(pos_lon_plus, star, observationTime) - calculateAltitude(pos_lon_minus, star, observationTime)) / (2 * h);

    return J;
}

double LocationDetermination::calculateAltitude(const Eigen::Vector2d &position, const ReferenceStarData &star, const std::chrono::system_clock::time_point &observationTime) {
    double lat = position.x();
    double lon = position.y();
    double dec = star.position.y();
    double ra = star.position.x();
    double st = siderealTime(observationTime);
    double ha = st - lon - ra;

    double sinAlt = std::sin(lat) * std::sin(dec) + std::cos(lat) * std::cos(dec) * std::cos(ha);
    double altitude = std::asin(std::max(-1.0, std::min(1.0, sinAlt)));

    // Improved atmospheric refraction correction
    double R = 0;
    if (altitude > -0.087) {
        double tan_z = std::tan(PI/2 - altitude);
        R = 0.0167 / tan_z - 0.00138 / (tan_z * tan_z * tan_z);
    }

    return altitude + R;
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

double LocationDetermination::calculateMeasuredAltitude(const Star &star, const CameraParameters &cameraParams) {
    double x = star.position.x - cameraParams.centerX;
    double y = star.position.y - cameraParams.centerY;
    double r = std::sqrt(x*x + y*y);
    double zenith_angle = std::atan2(r, cameraParams.focalLength);
    double altitude = PI / 2 - zenith_angle;
    
    // Apply atmospheric refraction correction
    double R = 0.0167 / std::tan(altitude + 0.0167 / (altitude + 0.0167));
    return altitude + R;
}

static std::vector<std::pair<Star, ReferenceStarData>> LocationDetermination::removeOutliers(
    const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars,
    const CameraParameters &cameraParams,
    const Eigen::Vector2d &initialGuess) {
    
    std::vector<std::pair<Star, ReferenceStarData>> filteredStars;
    std::vector<double> residuals;

    for (const auto &match : matchedStars) {
        double measuredAltitude = calculateMeasuredAltitude(match.first, cameraParams);
        double calculatedAltitude = calculateAltitude(initialGuess, match.second, std::chrono::system_clock::now());
        double residual = std::abs(measuredAltitude - calculatedAltitude);
        residuals.push_back(residual);
    }

    double median = calculateMedian(residuals);
    double mad = calculateMAD(residuals, median);
    double threshold = 2.5 * mad / 0.6745; // Assuming normal distribution

    for (size_t i = 0; i < matchedStars.size(); ++i) {
        if (residuals[i] <= threshold) {
            filteredStars.push_back(matchedStars[i]);
        }
    }

    return filteredStars;
}

double LocationDetermination::calculateMedian(std::vector<double> values) {
    size_t size = values.size();
    if (size == 0) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    if (size % 2 == 0) {
        return (values[size / 2 - 1] + values[size / 2]) / 2;
    } else {
        return values[size / 2];
    }
}

double LocationDetermination::calculateMAD(const std::vector<double>& values, double median) {
    std::vector<double> deviations;
    for (double value : values) {
        deviations.push_back(std::abs(value - median));
    }
    return calculateMedian(deviations);
}

double LocationDetermination::calculateMedian(std::vector<double> values) {
    size_t size = values.size();
    if (size == 0) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    if (size % 2 == 0) {
        return (values[size / 2 - 1] + values[size / 2]) / 2;
    } else {
        return values[size / 2];
    }
}

double LocationDetermination::calculateMAD(const std::vector<double>& values, double median) {
    std::vector<double> deviations;
    for (double value : values) {
        deviations.push_back(std::abs(value - median));
    }
    return calculateMedian(deviations);
}