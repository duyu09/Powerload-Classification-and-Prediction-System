#pragma once
#include "../sqlite/sqlite3.h"
#include "../log/log.h"
#include <memory>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <string>

class DataBase {
public:
    // Initialize the database singleton
    static void init(const char* path);

    // Get the singleton instance of the database
    static DataBase* GetInstance(); 

    // Insert a new user into the database
    bool insert_user(std::string user_password, std::string user_phone_number, std::string user_room); 

    // Query user password from the database
    std::string query_user_password(std::string user_phone_number); 

    // Query user ID from the database
    int64_t query_user_id(std::string user_phone_number); 

    // Query recent real-time data from the database
    std::deque<int64_t> query_real(int64_t user_id); 

    // Query recent prediction data from the database
    std::deque<int64_t> query_prediction(int64_t user_id); 

    // Query the legality status of the user from the database
    std::string query_legal(int64_t user_id); 

    // Constructor
    DataBase(const char* path); 

    // Destructor
    ~DataBase(); 

    // Singleton instance pointer
    static std::shared_ptr<DataBase> single_; 
private:
    // SQLite database handle
    sqlite3* handle_;

    // Mutex for thread safety
    std::mutex mtx_;
    
    // User ID 
    size_t user_id; 
};