#include "httpconnect.h"

const char* HttpConn::srcDir;
std::atomic<int> HttpConn::userCount;
bool HttpConn::isET;

HttpConn::HttpConn() { 
    fd_ = -1;
    my_addr_ = { 0 };
    isClosed_ = true;
};

HttpConn::~HttpConn() { 
    Close(); 
};

void HttpConn::init(int fd, const sockaddr_in& addr) {
    assert(fd > 0);
    userCount++;
    my_addr_ = addr;
    fd_ = fd;
    my_writeBuff_.Has_read_all();
    my_readBuff_.Has_read_all();
    isClosed_ = false;
    LOG_INFO("The %d client in, IP:port(%s:%d), userCount:%d", fd_, GetIP(), GetPort(), (int)userCount);
}

void HttpConn::Close() {
    if(isClosed_ == false){
        isClosed_ = true; 
        userCount--;
        close(fd_);
        LOG_INFO("The %d client quit, IP:port(%s:%d), userCount:%d", fd_, GetIP(), GetPort(), (int)userCount);
    }
}

int HttpConn::GetFd() const {
    return fd_;
};

const char* HttpConn::GetIP() const {
    return inet_ntoa(my_addr_.sin_addr);
}

int HttpConn::GetPort() const {
    return my_addr_.sin_port;
}

ssize_t HttpConn::read(int* saveErrno) {
    ssize_t len = -1;
    do {
        len = my_readBuff_.Read(fd_, saveErrno);
        if (len <= 0) {
            break;
        }
    } while (isET);
    return len;
}

ssize_t HttpConn::write(int* saveErrno) {
    ssize_t len = -1;
    do {
        len = writev(fd_, iov_, iov_count_);
        if(len <= 0) {
            *saveErrno = errno;
            break;
        }
        if(iov_[0].iov_len + iov_[1].iov_len  == 0) { break; } /* 传输结束 */
        else if(static_cast<size_t>(len) > iov_[0].iov_len) {
            iov_[1].iov_base = (uint8_t*) iov_[1].iov_base + (len - iov_[0].iov_len);
            iov_[1].iov_len -= (len - iov_[0].iov_len);
            if(iov_[0].iov_len) {
                my_writeBuff_.Has_read_all();
                iov_[0].iov_len = 0;
            }
        }
        else {
            iov_[0].iov_base = (uint8_t*)iov_[0].iov_base + len; 
            iov_[0].iov_len -= len; 
            my_writeBuff_.Has_read(len);
        }
    } while(isET || iov_[0].iov_len + iov_[1].iov_len > 0);
    return len;
}

bool HttpConn::process() {
    request_.Init();
    if(my_readBuff_.Get_readable_size() <= 0) {
        return false;
    }
    else if(request_.parse(my_readBuff_)) {
        response_.Init(200, request_.IsKeepAlive());
    } else {
        response_.Init(400, false);
    }

    response_.MakeResponse(my_writeBuff_, request_);

    iov_[0].iov_base = const_cast<char*>(my_writeBuff_.Get_read_point_const());
    iov_[0].iov_len = my_writeBuff_.Get_readable_size();
    iov_count_ = 1;

    return true;
}