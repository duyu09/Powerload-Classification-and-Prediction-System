#pragma once

#include <unordered_map>
#include <string>
#include <regex>
#include <errno.h>
#include "../buffer/buffer.h"
#include "../log/log.h"

class HttpRequest {
public:
    // Enumeration for parsing state
    enum PARSE_STATE {
        REQUEST_LINE,
        HEADERS,
        BODY,
        FINISH,
    };

    // Constructor
    HttpRequest() { Init(); }

    // Destructor
    ~HttpRequest() = default;

    // Initialize the HTTP request object
    void Init();

    // Parse the HTTP request from a buffer
    bool parse(Buffer& buff);

    // Get the path of the request
    std::string path() const;

    // Get the path of the request (mutable)
    std::string& path();

    // Check if the connection is keep-alive
    bool IsKeepAlive() const;

    /* 
    todo 
    void HttpConn::ParseFormData() {}
    void HttpConn::ParseJson() {}
    */
    
    // Map to store request parameters
    std::unordered_map<std::string, std::string> parameter;

private:
    // Parse the request line
    bool ParseRequestLine_(const std::string& line);

    // Parse a header line
    void ParseHeader_(const std::string& line);

    // Parse the request body
    void ParseBody_();

    // Parsing state
    PARSE_STATE state_;

    // Request method
    std::string method_;

    // Request path
    std::string path_;

    // Request HTTP version
    std::string version_;

    // Request headers
    std::unordered_map<std::string, std::string> header_;

    // Request body
    std::string body_;
};