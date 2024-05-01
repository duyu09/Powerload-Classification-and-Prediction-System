#pragma once

#include <vector>
#include <sys/epoll.h>
#include <unistd.h>
#include <stdexcept>
#include <fcntl.h>
#include <cassert>
#include <cerrno>

class Epoller {
public:
    // Constructor with optional capacity parameter
    explicit Epoller(int capacity = 1024);

    // Destructor
    ~Epoller();

    // Add a file descriptor to the epoll instance
    bool AddFileDescriptor(int fd, int events);

    // Modify the events associated with a file descriptor in the epoll instance
    bool ModifyFileDescriptor(int fd, int events);

    // Remove a file descriptor from the epoll instance
    bool RemoveFileDescriptor(int fd);

    // Wait for events on file descriptors registered with the epoll instance
    int WaitEvents(int timeoutMs = -1);

    // Get the file descriptor associated with an event at a given index
    int GetEventFileDescriptor(size_t index) const;

    // Get the event types associated with an event at a given index
    int GetEventTypes(size_t index) const;
        
private:
    // Modify a file descriptor in the epoll instance
    bool ModifyDescriptor(int operation, int fd, int events, epoll_event* ev);

    // File descriptor of the epoll instance
    int epoll_fd_;

    // Vector to store epoll events   
    std::vector<struct epoll_event> events_;
};