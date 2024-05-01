#pragma once

#include <unistd.h>
#include <sys/uio.h>
#include <cerrno>
#include <cstddef>
#include <sys/types.h>
#include <vector>
#include <string>

class Buffer {
public:
    // Constructor
    Buffer(size_t capacity = 1024);

    // Destructor
    ~Buffer() = default;

    // Get the capacity of the buffer
    size_t Get_capacity() const;

    // Get the size of readable data in the buffer
    size_t Get_readable_size() const;

    // Get the size of writable space in the buffer
    size_t Get_writable_size() const;

    // Get the size of data already read from the buffer
    size_t Get_already_read_size() const;

    // Get a constant pointer to the read point in the buffer
    const char* Get_read_point_const() const;

    // Get a pointer to the read point in the buffer
    char* Get_read_point();

    // Get a constant pointer to the write point in the buffer
    const char* Get_write_point_const() const;

    // Get a pointer to the write point in the buffer
    char* Get_write_point();

    // Move the write point forward by a specified size
    void Has_write(size_t size);

    // Move the read point forward by a specified size
    void Has_read(size_t size);

    // Reset the read and write points to the beginning of the buffer
    void Has_read_all();

    // Move the read point to a specific position in the buffer
    void Has_read_until(const char* ver);

    // Get all the readable data from the buffer and clear it
    std::string Get_read_all();

    // Append data to the buffer
    void Append(const char* data, size_t size);
    void Append(const std::string& data);
    
    // Read data from a file descriptor into the buffer
    ssize_t Read(int fd, int* err);

    // Write data from the buffer to a file descriptor
    ssize_t Write(int fd, int* err);

private:
    // Get a constant pointer to the start of the buffer
    const char* Get_start_point_const_() const;

    // Get a pointer to the start of the buffer
    char* Get_start_point_();

    // Ensure that there is enough writable space in the buffer
    void Ensure_writable_(size_t size);

    // Private member variables
    size_t start_read_;
    size_t start_write_;
    size_t capacity_;
    std::vector<char> buffer_;
};
