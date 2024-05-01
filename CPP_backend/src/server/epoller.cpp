#include "epoller.h"

Epoller::Epoller(int capacity): epoll_fd_(epoll_create(512)), events_(capacity) {
    if(epoll_fd_ < 0) {
        throw std::runtime_error("Failed to create epoll instance!");
    }
}

Epoller::~Epoller() {
    close(epoll_fd_);
}

bool Epoller::AddFileDescriptor(int fd, int events) {
    if(fd < 0) return false;
    epoll_event ev = {};
    ev.data.fd = fd;
    ev.events = events;
    return ModifyDescriptor(EPOLL_CTL_ADD, fd, events, &ev);
}

bool Epoller::ModifyFileDescriptor(int fd, int events) {
    if(fd < 0) return false;
    epoll_event ev = {};
    ev.data.fd = fd;
    ev.events = events;
    return ModifyDescriptor(EPOLL_CTL_MOD, fd, events, &ev);
}

bool Epoller::RemoveFileDescriptor(int fd) {
    if(fd < 0) return false;
    return ModifyDescriptor(EPOLL_CTL_DEL, fd, 0, nullptr);
}

bool Epoller::ModifyDescriptor(int operation, int fd, int events, epoll_event* ev) {
    return epoll_ctl(epoll_fd_, operation, fd, ev) == 0;
}

int Epoller::WaitEvents(int timeoutMs) {
    return epoll_wait(epoll_fd_, &events_[0], static_cast<int>(events_.size()), timeoutMs);
}

int Epoller::GetEventFileDescriptor(size_t index) const {
    if(index >= events_.size()) {
        throw std::out_of_range("Index out of range!");
    }
    return events_[index].data.fd;
}

int Epoller::GetEventTypes(size_t index) const {
    if(index >= events_.size()) {
        throw std::out_of_range("Index out of range!");
    }
    return events_[index].events;
}