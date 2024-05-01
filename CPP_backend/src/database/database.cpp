#include "database.h"

std::shared_ptr<DataBase> DataBase::single_ = nullptr;

void DataBase::init(const char* path) {
    if (!single_) {
        single_ = std::move(std::make_shared<DataBase>(path));
    }
}

DataBase* DataBase::GetInstance() {
    return single_.get();
}

DataBase::DataBase(const char* path) {
    sqlite3* db;
    if (sqlite3_open(path, &db) != SQLITE_OK) {
        LOG_ERROR("Error opening database.");
    } else {
        LOG_INFO("Opening database successful! The path is %s", path);
    }
    handle_ = db;
}

DataBase::~DataBase() {
    std::lock_guard<std::mutex> lock(mtx_);
    sqlite3_close(handle_);
}

bool DataBase::insert_user(std::string user_password, std::string user_phone_number, std::string user_room) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream sql;
    sql << "INSERT INTO user (user_id, user_password, user_phone_number, user_powermeter_id, user_room) VALUES (";
    sql << "(SELECT MAX(user_id) + 1 FROM user)" << ", ";
    sql << "?1, ";
    sql << "?2, ";
    sql << "0, ";
    sql << "?3);";

    sqlite3_stmt* statement;

    if (sqlite3_prepare_v2(handle_, sql.str().c_str(), -1, &statement, nullptr) == SQLITE_OK) {
        // if(sqlite3_bind_int(statement, 0, user_id) != SQLITE_OK) LOG_INFO("BIND_INT_0_ERROR");
        if(sqlite3_bind_text(statement, 1, user_password.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) LOG_INFO("BIND_TEXT_1_ERROR");
        if(sqlite3_bind_text(statement, 2, user_phone_number.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) LOG_INFO("BIND_TEXT_2_ERROR");
      // if(sqlite3_bind_int(statement, 3, 0) != SQLITE_OK) LOG_INFO("BIND_INT_3_ERROR");
        if(sqlite3_bind_text(statement, 3, user_room.c_str(), -1, SQLITE_STATIC) != SQLITE_OK) LOG_INFO("BIND_TEXT_3_ERROR");
        
        if (sqlite3_step(statement) != SQLITE_DONE) {
            LOG_ERROR("Error inserting data.");
        } else {
            LOG_INFO("Data inserted.");
        }

        sqlite3_finalize(statement);
        return true;
    }
    return false;
}

std::string DataBase::query_user_password(std::string user_phone_number) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream sql;
    
    sql << "SELECT user_password FROM user WHERE user_phone_number=\"" << user_phone_number << "\";";
    sqlite3_stmt* statement;
    if (sqlite3_prepare_v2(handle_, sql.str().c_str(), -1, &statement, nullptr) == SQLITE_OK) {
        while (sqlite3_step(statement) == SQLITE_ROW) {
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
            LOG_DEBUG("The text is %s", text);
            return text;
        }

        sqlite3_finalize(statement);
    }
    return "";
}

int64_t DataBase::query_user_id(std::string user_phone_number) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream sql;

    sql << "SELECT user_id FROM user WHERE user_phone_number = \"" << user_phone_number << "\";";

    sqlite3_stmt* statement;
    int64_t user_id = -1;
    if (sqlite3_prepare_v2(handle_, sql.str().c_str(), -1, &statement, nullptr) == SQLITE_OK) {
        if (sqlite3_step(statement) == SQLITE_ROW) {
            user_id = sqlite3_column_int64(statement, 0);
        }

        sqlite3_finalize(statement);
    }
    return user_id;
}

std::deque<int64_t> DataBase::query_real(int64_t user_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream sql;
    sql << "SELECT powerload_value FROM realtime_data WHERE user_id = " << user_id;
    sql << " ORDER BY time_unixstamp DESC LIMIT 50";

    sqlite3_stmt* statement;
    std::deque<int64_t> ans;
    if (sqlite3_prepare_v2(handle_, sql.str().c_str(), -1, &statement, nullptr) == SQLITE_OK) {
        while (sqlite3_step(statement) == SQLITE_ROW) {
            ans.push_front(sqlite3_column_int64(statement, 0) / 1000);
        }

        sqlite3_finalize(statement);
    }
    while(ans.size() < 50) {
        ans.push_front(0);
    }
    return ans;
}

std::deque<int64_t> DataBase::query_prediction(int64_t user_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream sql;
    sql << "SELECT pred_value FROM prediction_data WHERE user_id = " << user_id;
    sql << " ORDER BY time_unixstamp DESC LIMIT 20";
    
    sqlite3_stmt* statement;
    std::deque<int64_t> ans;
    if (sqlite3_prepare_v2(handle_, sql.str().c_str(), -1, &statement, nullptr) == SQLITE_OK) {
        while (sqlite3_step(statement) == SQLITE_ROW) {
            ans.push_front(sqlite3_column_int64(statement, 0) / 1000);
        }

        sqlite3_finalize(statement);
    }
    while(ans.size() < 20) {
        ans.push_back(0);
    }
    return ans;
}

std::string DataBase::query_legal(int64_t user_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream sql;
    sql << "SELECT label_value FROM label_data WHERE user_id = " << user_id;
    sql << " ORDER BY time_unixstamp DESC LIMIT 1";

    sqlite3_stmt* statement;
    std::string ans = "";

    if (sqlite3_prepare_v2(handle_, sql.str().c_str(), -1, &statement, nullptr) == SQLITE_OK) {
        while (sqlite3_step(statement) == SQLITE_ROW) {
            const char* text = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
            ans += text;
        }

        sqlite3_finalize(statement);
    } else {
        ans += "正在计算"; // Calculating in progress
    }
    if (ans == "") ans += "正在计算..."; // Calculating in progress
    return ans;
}
