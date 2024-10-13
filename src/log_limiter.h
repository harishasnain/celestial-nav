#pragma once
#include <unordered_map>
#include <string>

class LogLimiter {
public:
    static bool shouldLog(const std::string& key, int maxEntries = 10000) {
        static std::unordered_map<std::string, int> logCounts;
        if (++logCounts[key] <= maxEntries) {
            return true;
        }
        if (logCounts[key] == maxEntries + 1) {
            return true; // Log one last time to indicate that future logs will be suppressed
        }
        return false;
    }
};

// Helper function for string concatenation
inline std::string concat(const std::string& a, const std::string& b) {
    return a + b;
}
