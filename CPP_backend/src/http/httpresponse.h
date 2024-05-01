#pragma once

#include <unordered_map>
#include <fcntl.h>
#include <unistd.h> 
#include <sys/stat.h> 
#include "httprequest.h"
#include "../buffer/buffer.h"
#include "../database/database.h"
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>

class HttpResponse {
public:
    // Constructor
    HttpResponse();

    // Destructor
    ~HttpResponse() = default;

    // Initialize the HTTP response object
    void Init(size_t code_, bool isKeepAlive_);

    // Generate the HTTP response based on the request
    void MakeResponse(Buffer& buff, HttpRequest request);
    
private:
    // Handle login request
    void login_(Buffer &buff, HttpRequest request);

    // Handle register request
    void register_(Buffer &buff, HttpRequest request);

    // Handle getdata request
    void getdata_(Buffer &buff, HttpRequest request);

    // HTTP response status code
    int code_;

    // Flag indicating whether the connection should be kept alive
    bool isKeepAlive_;

    // Map to store matched patterns
    std::unordered_map<std::string, std::string> match_;

    // Map of HTTP status codes and their corresponding status messages
    const std::unordered_map<int, std::string> CODE_STATUS = {
        { 200, "OK" },
        { 400, "Bad Request" },
        { 403, "Forbidden" },
        { 404, "Not Found" },
    };

    // Map of HTTP status codes and their corresponding error page paths
    const std::unordered_map<int, std::string> CODE_PATH = {
        { 400, "/400.html" },
        { 403, "/403.html" },
        { 404, "/404.html" },
    };

    // Map of request paths and their corresponding actions
    const std::unordered_map<std::string, std::string> WEB_SITE = {
        {"/api/login", "LOGIN"},
        {"/api/register", "REGISTER"},
        {"/api/getdata", "GET"}
    };
};

class GUIDGenerator {
public:
    static std::string GenerateFakeGUID() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, std::numeric_limits<int>::max());

        std::stringstream ss;
        ss << std::hex << std::setw(8) << std::setfill('0') << dis(gen);
        ss << "";
        ss << std::setw(4) << std::setfill('0') << (dis(gen) & 0xFFFF);
        ss << "";
        ss << std::setw(4) << std::setfill('0') << ((dis(gen) & 0x0FFF) | 0x4000);
        ss << "";
        ss << std::setw(4) << std::setfill('0') << ((dis(gen) & 0x3FFF) | 0x8000);
        ss << "";
        ss << std::setw(12) << std::setfill('0') << dis(gen);
        return ss.str();
    }
};