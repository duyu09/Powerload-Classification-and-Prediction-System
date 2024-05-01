#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <cassert>
#include <sys/stat.h>
#include "blockqueue.h"
#include "../buffer/buffer.h"
#include <cstdarg>
#include <sys/select.h>

// Class for representing date
class Date {
public:
    // Constructor
    Date() : year(0), month(0), day(0), hour(0), minute(0), second(0) {}
    ~Date() = default;
    // Set date based on tm struct
    void Set(std::tm* ptm) {
        year = ptm->tm_year + 1900;
        month = ptm->tm_mon + 1;
        day = ptm->tm_mday;
        hour = ptm->tm_hour;
        minute = ptm->tm_min;
        second = ptm->tm_sec;
    }
    // Date components
    size_t year, month, day, hour, minute, second;
};

// Class for handling UTC timer
class UtcTimer {
public:
    // Constructor
    UtcTimer() {
        tm_ = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(tm_);
        std::tm* ptm = std::localtime(&tt);
        date_.Set(ptm);
        std::snprintf(utc_fmt_, 20, "%zu-%02zu-%02zu %02zu:%02zu:%02zu", date_.year, date_.month, date_.day, date_.hour, date_.minute, date_.second);
    }
    // Get current time as Date object
    auto get_cur_time() -> Date {
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - tm_);
        if (duration > std::chrono::seconds(1)) {
            tm_ = now;
            date_.second += static_cast<size_t>(duration.count());
            if (date_.second >= 60) {
                tm_ = now;
                std::time_t tt = std::chrono::system_clock::to_time_t(tm_);
                std::tm* ptm = std::localtime(&tt);
                std::snprintf(utc_fmt_, 20, "%zu-%02zu-%02zu %02zu:%02zu:%02zu", date_.year, date_.month, date_.day, date_.hour, date_.minute, date_.second);
            } else {
                std::snprintf(utc_fmt_ + 17, 3, "%02zu", date_.second);
            }
        }
        return date_;
    }
    // Get current time as string
    std::string get_cur_time_str() {
        get_cur_time();
        return std::string(utc_fmt_);
    }
private:
    std::chrono::system_clock::time_point tm_;
    Date date_;
    char utc_fmt_[20];
};

// Class for logging
class Log {
public:
    static Log* getInstance();
    // Initialize logging system
    void Init(size_t level = 1,
              const char* path = "../work_log",
              const char* suffix = ".log",
              size_t capacity = 0);
    static void Flush_all();
    // Write log message
    void Write(size_t level, const char* format, ...);
    // Flush logs
    void Flush();
    // Get logging level
    size_t Get_level();
    // Set logging level
    void Set_level(size_t level);
    // Check if logging is open
    bool Is_open();
    Log() = default;
    ~Log() = default;
private:
    // Append log level to buffer
    void Append_log_level_(size_t level);
    // Asynchronously write logs
    void Async_write_();
    // Initialize log file
    void Init_file_();
    // Close log file
    void Close_file_();

    static constexpr int NAME_LEN_ = 256;
    static constexpr int MAX_LINE_NUMBER_ = 50000;

    const char* path_;
    const char* suffix_;

    int line_count_;
    int to_day_;

    bool is_open_;

    Buffer buffer_;
    int level_;
    bool is_async_;

    UtcTimer tm_;

    FILE* fp_;
    std::unique_ptr<BlockQueue<std::string>> q_;
    std::unique_ptr<std::thread> write_thread_;
    std::mutex mtx_;
};

// Macro for logging debug information
#define LOG_BASE(level, format, ...) \
    do {\
        Log* log = Log::getInstance();\
        if (log->Is_open() && log->Get_level() <= level) {\
            log->Write(level, format, ##__VA_ARGS__); \
            log->Flush();\
        }\
    } while(0);

#define LOG_DEBUG(format, ...) do {LOG_BASE(0, format, ##__VA_ARGS__)} while(0);
#define LOG_INFO(format, ...) do {LOG_BASE(1, format, ##__VA_ARGS__)} while(0);
#define LOG_WARN(format, ...) do {LOG_BASE(2, format, ##__VA_ARGS__)} while(0);
#define LOG_ERROR(format, ...) do {LOG_BASE(3, format, ##__VA_ARGS__)} while(0);