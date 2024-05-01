#pragma once

#include <sys/types.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <stdlib.h> 
#include <atomic>
#include "../buffer/buffer.h"
#include "httprequest.h"
#include "httpresponse.h"

class HttpConn {
public:
    // Constructor
    HttpConn(); 

    // Destructor
    ~HttpConn(); 

    // Initialize connection
    void init(int sockFd, const sockaddr_in& addr); 

    // Read data from the connection
    ssize_t read(int* saveErrno); 

    // Write data to the connection
    ssize_t write(int* saveErrno); 

    // Close the connection
    void Close(); 

    // Get the file descriptor of the connection
    int GetFd() const; 

    // Get the port of the connection
    int GetPort() const; 

    // Get the IP address of the connection
    const char* GetIP() const; 

    // Process the request
    bool process(); 

    // Static member variables
    static bool isET; // Flag for edge-triggered mode
    static const char* srcDir; // Source directory
    static std::atomic<int> userCount; // Atomic counter for user connections

private:
    int fd_; // File descriptor of the connection
    struct sockaddr_in my_addr_; // Address structure of the connection
    bool isClosed_; // Flag indicating if the connection is closed
    int iov_count_; // Number of IO vectors
    struct iovec iov_[2]; // IO vectors for reading and writing
    Buffer my_readBuff_; // Read buffer for incoming data
    Buffer my_writeBuff_; // Write buffer for outgoing data
    HttpRequest request_; // HTTP request object
    HttpResponse response_; // HTTP response object
};
