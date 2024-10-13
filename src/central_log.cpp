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

    void customLog(LogLevel level, const std::string& file, int line, const std::string& function, const std::string& message) {
        if (logFile_.is_open()) {
            logFile_ << getCurrentTimestamp() << " [" << getLevelString(level) << "] "
                     << file << ":" << line << " " << function << " - " << message << std::endl;
        }
    }

private:
    CentralLog() {
        const std::string logFilePath = "/home/haris/celestial-nav/log.txt";
        logFile_.open(logFilePath, std::ios::app);
        if (!logFile_.is_open()) {
            std::cerr << "Error: Unable to open log file at " << logFilePath << std::endl;
        }
    }

    ~CentralLog() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    std::ofstream logFile_;

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
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
    if (LogLimiter::shouldLog(__FUNCTION__)) { \
        CentralLog::getInstance().customLog(level, __FILE__, __LINE__, __FUNCTION__, message); \
    } else if (LogLimiter::shouldLog(std::string(__FUNCTION__) + "_suppressed", 1)) { \
        CentralLog::getInstance().customLog(level, __FILE__, __LINE__, __FUNCTION__, "Further logs from this function will be suppressed"); \
    }

#define LOG_DEBUG(message) LOG(CentralLog::LogLevel::DEBUG, message)
#define LOG_INFO(message) LOG(CentralLog::LogLevel::INFO, message)
#define LOG_WARNING(message) LOG(CentralLog::LogLevel::WARNING, message)
#define LOG_ERROR(message) LOG(CentralLog::LogLevel::ERROR, message)
