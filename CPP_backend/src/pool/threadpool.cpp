#include "threadpool.h"

ThreadPool::ThreadPool(size_t thread_num) {
    for(size_t i = 0; i < thread_num; ++i) {
        threads_.emplace_back(std::thread(&ThreadPool::Running, this));
    }
}

ThreadPool::~ThreadPool() {
    std::unique_lock<std::mutex> lock(mtx_);
    is_closed_ = true;
    lock.unlock();
    cond_.notify_all();
    for(auto& thread : threads_) {
        if(thread.joinable()) {
            thread.join();
        }
    }
}

void ThreadPool::Running() {
    while(true) {
        std::function<void()> task;
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [this]{ return !tasks_.empty() || is_closed_; });
        if(is_closed_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
        lock.unlock();
        task();
    }
}