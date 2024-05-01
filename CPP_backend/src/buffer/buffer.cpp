#include "buffer.h"

Buffer::Buffer(size_t capacity): capacity_(capacity), 
        start_read_(0), 
        start_write_(0), 
        buffer_(capacity, 0) {}

size_t Buffer::Get_capacity() const {
    return capacity_;
}
    
size_t Buffer::Get_readable_size() const {
    return start_write_ - start_read_;
}

size_t Buffer::Get_writable_size() const {
    return capacity_ - start_write_;
}

size_t Buffer::Get_already_read_size() const {
    return start_read_;
}

const char * Buffer::Get_read_point_const() const {
    return Get_start_point_const_() + start_read_;
}

char * Buffer::Get_read_point() {
    return Get_start_point_() + start_read_;
}

const char * Buffer::Get_write_point_const() const {
    return Get_start_point_const_() + start_write_;
}

char * Buffer::Get_write_point() {
    return Get_start_point_() + start_write_;
}

void Buffer::Has_write(size_t size) {
    start_write_ += size;
}

void Buffer::Has_read(size_t size) {
    start_read_ += size;
}

void Buffer::Has_read_all() {
    std::fill(Get_start_point_(), Get_start_point_() + start_write_, 0);
    start_read_ = start_write_ = 0;
}

void Buffer::Has_read_until(const char *ver) {
    Has_read(ver - Get_read_point());
}

std::string Buffer::Get_read_all() {
    std::string str = std::string(Get_read_point_const(), Get_readable_size());
    Has_read_all();
    return str;
}

void Buffer::Append(const char *data, size_t size) {
    Ensure_writable_(size);
    std::copy(data, data + size, Get_write_point());
    Has_write(size);
}

void Buffer::Append(const std::string &data) {
    Append(data.data(), data.size());
}
    
ssize_t Buffer::Read(int fd, int *err) {
    char buff[65535];
    struct iovec iov[2];
    const size_t writeable_size = Get_writable_size();
    iov[0].iov_base = Get_write_point();
    iov[0].iov_len = writeable_size;
    iov[1].iov_base = buff;
    iov[1].iov_len = sizeof(buff);
    ssize_t len = readv(fd, iov, 2);
    if(len < 0) {
        *err = errno;
    } else if(static_cast<size_t>(len) <= writeable_size) {
        start_write_ += len;
    } else {
        start_write_ = buffer_.size();
        Append(buff, len - writeable_size);
    }
    return len;
}

ssize_t Buffer::Write(int fd, int *err) {
    size_t size = Get_readable_size();
    ssize_t len = write(fd, Get_write_point(), size);
    if(len < 0) {
        *err = errno;
    }
    return len;
}

const char * Buffer::Get_start_point_const_() const {
    return buffer_.data();
}

char * Buffer::Get_start_point_(){
    return &*buffer_.begin();
}

void Buffer::Ensure_writable_(size_t size) {
    if(Get_writable_size() >= size) {
        return ;
    }
    if(Get_writable_size() + Get_already_read_size() >= size) {
        std::copy(Get_read_point_const(), Get_write_point_const(), Get_start_point_());
        start_write_ = start_write_ - Get_already_read_size();
        start_read_ = 0;
    } else {
        std::copy(Get_read_point_const(), Get_write_point_const(), Get_start_point_());
        start_write_ = start_write_ - Get_already_read_size();
        start_read_ = 0;
        buffer_.resize(size + start_write_);
    }
}