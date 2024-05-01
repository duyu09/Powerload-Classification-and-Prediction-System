#pragma once

#include <assert.h>
#include <sys/socket.h>
#include <cstring>
#include "epoller.h"
#include "../log/log.h"
#include "../timer/timer.h"
#include "../pool/threadpool.h"
#include "../http/httpconnect.h"

class WebServer {
public:
    // Constructor
    WebServer(int port, int trigMode, int timeoutMS, bool OptLinger, 
        const char* dbPath, int connPoolNum, int threadNum,
        bool openLog, int logLevel, int logQueSize);

    // Destructor
    ~WebServer();

    // Start the web server
    void Start();

private:
    // Initialize the socket
    bool InitSocket_(); 

    // Initialize event mode based on the trigger mode
    void InitEventMode_(int trigMode);

    // Add client connection to the server
    void AddClient_(int fd, sockaddr_in addr);
  
    // Deal with the listening socket
    void DealListen_();

    // Deal with write events for a client
    void DealWrite_(HttpConn* client);

    // Deal with read events for a client
    void DealRead_(HttpConn* client);

    // Send error message to client
    void SendError_(int fd, const char*info);

    // Extend timeout for a client connection
    void ExtentTime_(HttpConn* client);

    // Close client connection
    void CloseConn_(HttpConn* client);

    // Callback function for read events
    void OnRead_(HttpConn* client);

    // Callback function for write events
    void OnWrite_(HttpConn* client);

    // Process HTTP request from client
    void OnProcess(HttpConn* client);

    // Set file descriptor to non-blocking mode
    static int SetFdNonblock(int fd);

    // Maximum file descriptor
    static const int MAX_FD = 65536;

    // Port number
    int port_;

    // Whether to enable SO_LINGER option
    bool openLinger_;

    // Timeout in milliseconds
    int timeoutMS_;

    // Flag to indicate if server is closed
    bool isClose_;

    // Listening socket file descriptor
    int listenFd_;
    
    // Event type for listening socket
    uint32_t listenEvent_;

    // Event type for client connections
    uint32_t connEvent_;
   
    // Timer for managing client connections
    std::unique_ptr<Timer> timer_;

    // Thread pool for handling client requests
    std::unique_ptr<ThreadPool> threadpool_;

    std::unique_ptr<Epoller> epoller_;

    std::unordered_map<int, HttpConn> users_;
};
