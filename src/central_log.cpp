#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>
#include "log_limiter.h"

class CentralLog {
public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    static CentralLog& getInstance() {
        static CentralLog instance;
        return instance;
    }

    void log(LogLevel level, const std::string& file, const std::string& function, int line, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            logFile_ << getCurrentTimestamp() << " [" << getLevelString(level) << "] "
                     << file << ":" << function << ":" << line << " - " << message << std::endl;
            logFile_.flush();  // Flush the buffer immediately
        } catch (const std::exception& e) {
            std::cerr << "Error writing to log file: " << e.what() << std::endl;
        }
    }

private:
    CentralLog() : logFile_("/home/haris/celestial-nav/src/central_log.txt", std::ios::app) {
        if (!logFile_.is_open()) {
            throw std::runtime_error("Unable to open log file");
        }
    }
    ~CentralLog() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    std::ofstream logFile_;
    std::mutex mutex_;

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    std::string getLevelString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

#define LOG(level, message) \
    CentralLog::getInstance().log(level, __FILE__, __FUNCTION__, __LINE__, message)

#define LOG_DEBUG(message) { \
    if (LogLimiter::shouldLog(__FUNCTION__)) { \
        log("DEBUG", __FILE__, __LINE__, __FUNCTION__, message); \
    } else if (LogLimiter::shouldLog(__FUNCTION__ "_suppressed", 1)) { \
        log("DEBUG", __FILE__, __LINE__, __FUNCTION__, "Further logs from this function will be suppressed"); \
    } \
}

#define LOG_INFO(message) { \
    if (LogLimiter::shouldLog(__FUNCTION__)) { \
        log("INFO", __FILE__, __LINE__, __FUNCTION__, message); \
    } else if (LogLimiter::shouldLog(__FUNCTION__ "_suppressed", 1)) { \
        log("INFO", __FILE__, __LINE__, __FUNCTION__, "Further logs from this function will be suppressed"); \
    } \
}

#define LOG_WARNING(message) LOG(CentralLog::LogLevel::WARNING, message)
#define LOG_ERROR(message) LOG(CentralLog::LogLevel::ERROR, message)
