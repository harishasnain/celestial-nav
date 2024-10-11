#include "star_matching.h"
#include "location_determination.h"
#include "camera_parameters.h"

constexpr double PI = 3.14159265358979323846;

Eigen::Vector2d LocationDetermination::determineLocation(const std::vector<std::pair<Star, ReferenceStarData>> &matchedStars,
                                                         const std::chrono::system_clock::time_point &observationTime,
                                                         const CameraParameters &cameraParams,
                                                         const Eigen::Vector2d &initialGuess) {
    
    if (matchedStars.empty()) {
        std::cerr << "Error: No matched stars provided." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    Eigen::Vector2d position = initialGuess;
    if (position.x() == 0 && position.y() == 0) {
        position = calculateInitialGuess(matchedStars, observationTime);
    }
    std::cout << "Initial position: (" << position.x() << ", " << position.y() << ")" << std::endl;
    
    if (std::isnan(position.x()) || std::isnan(position.y())) {
        std::cerr << "Error: Initial position resulted in NaN values." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    // Ensure initial position is within valid ranges
    position.x() = std::max(-PI/2, std::min(PI/2, position.x()));
    position.y() = std::fmod(position.y() + 3*PI, 2*PI) - PI;
    
    auto filteredStars = removeOutliers(matchedStars, cameraParams, position);
    std::cout << "Removed " << matchedStars.size() - filteredStars.size() << " outliers." << std::endl;

    const int MAX_ITERATIONS = 100;
    const double POSITION_TOLERANCE = 1e-8;
    const double RESIDUAL_TOLERANCE = 1e-6;
    double prev_ssr = std::numeric_limits<double>::max();

    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        Eigen::MatrixXd J(filteredStars.size() * 2, 2);
        Eigen::VectorXd residuals(filteredStars.size() * 2);

        for (size_t i = 0; i < filteredStars.size(); ++i) {
            const auto &star = filteredStars[i].second;
            J.block<2, 2>(i*2, 0) = calculateJacobian(position, star, observationTime);
            
            double measuredAltitude = calculateMeasuredAltitude(filteredStars[i].first, cameraParams);
            double calculatedAltitude = calculateAltitude(position, star, observationTime);
            residuals(i*2) = measuredAltitude - calculatedAltitude;
            
            double measuredAzimuth = std::atan2(filteredStars[i].first.position.y(), filteredStars[i].first.position.x());
            double calculatedAzimuth = calculateAzimuth(position, star, observationTime);
            residuals(i*2 + 1) = measuredAzimuth - calculatedAzimuth;
            
            std::cout << "Star " << i << ": Measured altitude: " << measuredAltitude 
                      << ", Calculated altitude: " << calculatedAltitude
                      << ", Residual: " << residuals(i*2) << std::endl;
            std::cout << "Star " << i << ": Measured azimuth: " << measuredAzimuth 
                      << ", Calculated azimuth: " << calculatedAzimuth
                      << ", Residual: " << residuals(i*2 + 1) << std::endl;
        }

        Eigen::Vector2d delta = (J.transpose() * J).ldlt().solve(J.transpose() * residuals);
        position += delta;

        // Ensure position stays within valid ranges
        position.x() = std::max(-PI/2, std::min(PI/2, position.x()));
        position.y() = std::fmod(position.y() + 3*PI, 2*PI) - PI;

        double ssr = residuals.squaredNorm();
        std::cout << "Iteration " << iteration << " - SSR: " << ssr << std::endl;

        if (delta.norm() < POSITION_TOLERANCE || std::abs(prev_ssr - ssr) < RESIDUAL_TOLERANCE) {
            std::cout << "Converged after " << iteration + 1 << " iterations." << std::endl;
            break;
        }

        prev_ssr = ssr;
    }

    std::cout << "Final position: (" << position.x() << ", " << position.y() << ")" << std::endl;
    std::cout << "Final latitude (degrees): " << position.x() * 180 / PI << std::endl;
    std::cout << "Final longitude (degrees): " << position.y() * 180 / PI << std::endl;

    return position;
}

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars, const std::chrono::system_clock::time_point& observationTime) {
    if (matchedStars.size() < 3) {
        std::cerr << "Error: At least 3 matched stars are required for initial guess." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    // Select three brightest stars
    auto sortedStars = matchedStars;
    std::sort(sortedStars.begin(), sortedStars.end(),
        [](const auto& a, const auto& b) { return a.second.magnitude < b.second.magnitude; });

    const auto& star1 = sortedStars[0].second;
    const auto& star2 = sortedStars[1].second;
    const auto& star3 = sortedStars[2].second;

    // Calculate angular distances between stars
    double angle12 = angularDistance(star1.position, star2.position);
    double angle23 = angularDistance(star2.position, star3.position);
    double angle31 = angularDistance(star3.position, star1.position);

    // Use spherical trigonometry to estimate latitude and longitude
    double lat = estimateLatitude(angle12, angle23, angle31);
    double lon = estimateLongitude(star1.position, star2.position, star3.position, lat, siderealTime(observationTime));

    return Eigen::Vector2d(lat, lon);
}

double LocationDetermination::angularDistance(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2) {
    return std::acos(std::sin(star1.y()) * std::sin(star2.y()) +
                     std::cos(star1.y()) * std::cos(star2.y()) * std::cos(star1.x() - star2.x()));
}

double LocationDetermination::estimateLatitude(double angle12, double angle23, double angle31) {
    // Use the cosine rule for spherical triangles to estimate latitude
    double cosLat = (std::cos(angle12) + std::cos(angle23) * std::cos(angle31)) /
                    (std::sin(angle23) * std::sin(angle31));
    return std::acos(std::max(-1.0, std::min(1.0, cosLat)));
}

double LocationDetermination::estimateLongitude(const Eigen::Vector2d& star1, const Eigen::Vector2d& star2,
                                                const Eigen::Vector2d& star3, double lat, double lst) {
    // Use the position of the stars and estimated latitude to calculate longitude
    double ha1 = calculateHourAngle(star1, lat);
    double ha2 = calculateHourAngle(star2, lat);
    double ha3 = calculateHourAngle(star3, lat);

    double avgRA = (star1.x() + star2.x() + star3.x()) / 3.0;
    double avgHA = (ha1 + ha2 + ha3) / 3.0;

    return std::fmod(lst - avgRA + avgHA + 3*PI, 2*PI) - PI;
}

double LocationDetermination::calculateHourAngle(const Eigen::Vector2d& star, double lat) {
    double sinAlt = std::sin(star.y()) * std::sin(lat) + std::cos(star.y()) * std::cos(lat);
    double cosHA = (sinAlt - std::sin(star.y()) * std::sin(lat)) / (std::cos(star.y()) * std::cos(lat));
    return std::acos(std::max(-1.0, std::min(1.0, cosHA)));
}

Eigen::RowVector2d LocationDetermination::calculateJacobian(const Eigen::Vector2d& position, const ReferenceStarData& star, const std::chrono::system_clock::time_point& observationTime) {
    const double epsilon = 1e-6;
    Eigen::Vector2d positionPlusEpsilonLat(position.x() + epsilon, position.y());
    Eigen::Vector2d positionPlusEpsilonLon(position.x(), position.y() + epsilon);

    double f0 = calculateAltitude(position, star, observationTime);
    double fLat = calculateAltitude(positionPlusEpsilonLat, star, observationTime);
    double fLon = calculateAltitude(positionPlusEpsilonLon, star, observationTime);

    double dfdLat = (fLat - f0) / epsilon;
    double dfdLon = (fLon - f0) / epsilon;

    return Eigen::RowVector2d(dfdLat, dfdLon);
}

double LocationDetermination::calculateAltitude(const Eigen::Vector2d& position, const ReferenceStarData& star, const std::chrono::system_clock::time_point& observationTime) {
    double lat = position.x();
    double lon = position.y();
    double lst = siderealTime(observationTime);
    double alt, az;
    raDecToAltAz(star.position.x(), star.position.y(), lat, lon, lst, alt, az);
    return alt;
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

std::vector<std::pair<Star, ReferenceStarData>> LocationDetermination::removeOutliers(
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
    
    // Use a more adaptive threshold based on the number of stars
    double threshold = (matchedStars.size() > 50) ? 3.0 * mad / 0.6745 : 2.5 * mad / 0.6745;

    for (size_t i = 0; i < matchedStars.size(); ++i) {
        if (residuals[i] <= threshold) {
            filteredStars.push_back(matchedStars[i]);
        }
    }

    // Ensure we keep at least 20% of the stars
    if (filteredStars.size() < 0.2 * matchedStars.size()) {
        std::sort(residuals.begin(), residuals.end());
        size_t keepCount = static_cast<size_t>(0.2 * matchedStars.size());
        threshold = residuals[keepCount - 1];
        filteredStars.clear();
        for (size_t i = 0; i < matchedStars.size(); ++i) {
            if (residuals[i] <= threshold) {
                filteredStars.push_back(matchedStars[i]);
            }
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

double LocationDetermination::calculateGuessScore(const Eigen::Vector2d& guess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars) {
    double score = 0;
    for (int i = 0; i < numStars; ++i) {
        double calculatedAltitude = calculateAltitude(guess, stars[i].second, std::chrono::system_clock::now());
        double measuredAltitude = calculateMeasuredAltitude(stars[i].first, CameraParameters()); // You may need to pass proper camera parameters here
        score += std::pow(calculatedAltitude - measuredAltitude, 2);
    }
    return score;
}

Eigen::Vector2d LocationDetermination::nelderMeadOptimization(const Eigen::Vector2d& initialGuess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars) {
    const int maxIterations = 100;
    const double alpha = 1.0; // Reflection coefficient
    const double gamma = 2.0; // Expansion coefficient
    const double rho = 0.5;   // Contraction coefficient
    const double sigma = 0.5; // Shrink coefficient

    std::vector<Eigen::Vector2d> simplex = {
        initialGuess,
        initialGuess + Eigen::Vector2d(0.01, 0),
        initialGuess + Eigen::Vector2d(0, 0.01)
    };

    std::vector<double> scores = {
        calculateGuessScore(simplex[0], stars, numStars),
        calculateGuessScore(simplex[1], stars, numStars),
        calculateGuessScore(simplex[2], stars, numStars)
    };

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Sort vertices
        std::sort(simplex.begin(), simplex.end(),
                  [&](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                      return calculateGuessScore(a, stars, numStars) < calculateGuessScore(b, stars, numStars);
                  });

        // Calculate centroid of the best points
        Eigen::Vector2d centroid = (simplex[0] + simplex[1]) / 2.0;

        // Reflection
        Eigen::Vector2d reflected = centroid + alpha * (centroid - simplex[2]);
        double reflectedScore = calculateGuessScore(reflected, stars, numStars);

        if (reflectedScore < scores[0]) {
            // Expansion
            Eigen::Vector2d expanded = centroid + gamma * (reflected - centroid);
            double expandedScore = calculateGuessScore(expanded, stars, numStars);

            if (expandedScore < reflectedScore) {
                simplex[2] = expanded;
                scores[2] = expandedScore;
            } else {
                simplex[2] = reflected;
                scores[2] = reflectedScore;
            }
        } else if (reflectedScore < scores[1]) {
            simplex[2] = reflected;
            scores[2] = reflectedScore;
        } else {
            // Contraction
            Eigen::Vector2d contracted = centroid + rho * (simplex[2] - centroid);
            double contractedScore = calculateGuessScore(contracted, stars, numStars);

            if (contractedScore < scores[2]) {
                simplex[2] = contracted;
                scores[2] = contractedScore;
            } else {
                // Shrink
                simplex[1] = simplex[0] + sigma * (simplex[1] - simplex[0]);
                simplex[2] = simplex[0] + sigma * (simplex[2] - simplex[0]);
                scores[1] = calculateGuessScore(simplex[1], stars, numStars);
                scores[2] = calculateGuessScore(simplex[2], stars, numStars);
            }
        }

        // Check for convergence
        if ((simplex[0] - simplex[2]).norm() < 1e-6) {
            break;
        }
    }

    return simplex[0];
}

Eigen::Vector2d LocationDetermination::multiStageOptimization(const Eigen::Vector2d& initialGuess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars) {
    // Stage 1: Coarse grid search
    const int gridSize = 9;
    const double searchRadius = PI / 9; // 20 degrees
    Eigen::Vector2d bestGuess = initialGuess;
    double bestScore = calculateGuessScore(bestGuess, stars, numStars);

    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            double testLat = initialGuess.x() + (i - gridSize/2) * searchRadius / gridSize;
            double testLon = initialGuess.y() + (j - gridSize/2) * searchRadius / gridSize;
            
            testLat = std::max(-PI/2, std::min(PI/2, testLat));
            testLon = std::fmod(testLon + 3*PI, 2*PI) - PI;

            Eigen::Vector2d testGuess(testLat, testLon);
            double score = calculateGuessScore(testGuess, stars, numStars);
            if (score < bestScore) {
                bestScore = score;
                bestGuess = testGuess;
            }
        }
    }

    // Stage 2: Fine grid search
    const int fineGridSize = 5;
    const double fineSearchRadius = searchRadius / gridSize;

    for (int i = 0; i < fineGridSize; ++i) {
        for (int j = 0; j < fineGridSize; ++j) {
            double testLat = bestGuess.x() + (i - fineGridSize/2) * fineSearchRadius / fineGridSize;
            double testLon = bestGuess.y() + (j - fineGridSize/2) * fineSearchRadius / fineGridSize;
            
            testLat = std::max(-PI/2, std::min(PI/2, testLat));
            testLon = std::fmod(testLon + 3*PI, 2*PI) - PI;

            Eigen::Vector2d testGuess(testLat, testLon);
            double score = calculateGuessScore(testGuess, stars, numStars);
            if (score < bestScore) {
                bestScore = score;
                bestGuess = testGuess;
            }
        }
    }

    // Stage 3: Nelder-Mead optimization
    bestGuess = nelderMeadOptimization(bestGuess, stars, numStars);

    // Stage 4: Gradient descent fine-tuning
    const int maxGradientSteps = 50;
    const double learningRate = 0.01;
    const double gradientTolerance = 1e-8;

    for (int step = 0; step < maxGradientSteps; ++step) {
        Eigen::Vector2d gradient = calculateGradient(bestGuess, stars, numStars);
        double gradientNorm = gradient.norm();

        if (gradientNorm < gradientTolerance) {
            break;
        }

        bestGuess -= learningRate * gradient;
        bestGuess.x() = std::max(-PI/2, std::min(PI/2, bestGuess.x()));
        bestGuess.y() = std::fmod(bestGuess.y() + 3*PI, 2*PI) - PI;
    }

    return bestGuess;
}

Eigen::Vector2d LocationDetermination::calculateGradient(const Eigen::Vector2d& guess, const std::vector<std::pair<Star, ReferenceStarData>>& stars, int numStars) {
    const double h = 1e-6; // Small step for numerical differentiation
    Eigen::Vector2d gradient;

    for (int i = 0; i < 2; ++i) {
        Eigen::Vector2d guessPlus = guess;
        Eigen::Vector2d guessMinus = guess;
        guessPlus(i) += h;
        guessMinus(i) -= h;

        double scorePlus = calculateGuessScore(guessPlus, stars, numStars);
        double scoreMinus = calculateGuessScore(guessMinus, stars, numStars);

        gradient(i) = (scorePlus - scoreMinus) / (2 * h);
    }

    return gradient;
}

void LocationDetermination::raDecToAltAz(double ra, double dec, double lat, double lon, double lst, double& alt, double& az) {
    double ha = lst - ra;
    double sinAlt = sin(dec) * sin(lat) + cos(dec) * cos(lat) * cos(ha);
    alt = asin(sinAlt);
    double cosAz = (sin(dec) - sin(lat) * sinAlt) / (cos(lat) * cos(alt));
    az = acos(std::max(-1.0, std::min(1.0, cosAz)));
    if (sin(ha) > 0) {
        az = 2 * PI - az;
    }
}

double LocationDetermination::calculateAzimuth(const Eigen::Vector2d& position, const ReferenceStarData& star, const std::chrono::system_clock::time_point& observationTime) {
    double lat = position.x();
    double lon = position.y();
    double lst = siderealTime(observationTime);
    double alt, az;
    raDecToAltAz(star.position.x(), star.position.y(), lat, lon, lst, alt, az);
    return az;
}