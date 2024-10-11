#include "star_matching.h"
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
    std::cout << "Final latitude (degrees): " << position.x() * 180 / PI << std::endl;
    std::cout << "Final longitude (degrees): " << position.y() * 180 / PI << std::endl;
    return position;
}

Eigen::Vector2d LocationDetermination::calculateInitialGuess(const std::vector<std::pair<Star, ReferenceStarData>>& matchedStars) {
    if (matchedStars.empty()) {
        std::cerr << "Error: No matched stars available for initial guess." << std::endl;
        return Eigen::Vector2d(0, 0);
    }

    // Use the first two stars to estimate the position
    const auto& star1 = matchedStars[0].first;
    const auto& star2 = matchedStars[1].first;

    double az1, alt1, az2, alt2;
    raDecToAltAz(star1.ra, star1.dec, 0, 0, siderealTime(std::chrono::system_clock::now()), alt1, az1);
    raDecToAltAz(star2.ra, star2.dec, 0, 0, siderealTime(std::chrono::system_clock::now()), alt2, az2);

    double lat = (alt1 + alt2) / 2;
    double lon = (az1 + az2) / 2;

    return Eigen::Vector2d(lat, lon);
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