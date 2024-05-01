#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <functional>
#include <utility>

// Thread-safe block queue implementation
template<typename T>
class BlockQueue {
public:
    // Constructor
    BlockQueue(size_t capacity);

    // Destructor
    ~BlockQueue();

    // Clear the queue
    void Clear();

    // Check if the queue is empty
    bool Empty();

    // Check if the queue is full
    bool Full();

    // Get the size of the queue
    size_t Size();

    // Get the capacity of the queue
    size_t Capacity();

    // Get the front element of the queue
    T Front();

    // Push an element into the queue
    void Push(const T &item);

    // Pop an element from the queue
    bool Pop();

    // Flush the queue
    void Flush();
private:
    // Internal queue
    std::queue<T> q_;

    // Maximum capacity of the queue
    std::atomic<size_t> capacity_;      

    // Flag indicating whether the queue is closed
    std::atomic<bool> closed_;   

    // Mutex for thread safety
    std::mutex mtx_;             

    // Condition variable for consumers
    std::condition_variable consumer_;    
      
    // Condition variable for producers
    std::condition_variable producer_;      
};

template<typename T>
BlockQueue<T>::BlockQueue(size_t capacity): capacity_(capacity), closed_(false) {}

template<typename T>
BlockQueue<T>::~BlockQueue() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::queue<T> tmp;
    std::swap(tmp, q_);
    closed_ = true;
    consumer_.notify_all();
    producer_.notify_all();
}

template<typename T>
void BlockQueue<T>::Clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::queue<T> tmp;
    std::swap(tmp, q_);
}

template<typename T>
bool BlockQueue<T>::Empty() {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.empty();
}

template<typename T>
bool BlockQueue<T>::Full() {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.size() == capacity_;
}

template<typename T>
size_t BlockQueue<T>::Size() {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.size();
}

template<typename T>
size_t BlockQueue<T>::Capacity() {
    std::lock_guard<std::mutex> lock(mtx_);
    return capacity_;
}

template<typename T>
T BlockQueue<T>::Front() {
    std::lock_guard<std::mutex> lock(mtx_);
    return q_.front();
}

template<typename T>
void BlockQueue<T>::Push(const T &item) {
    std::unique_lock<std::mutex> lock(mtx_);
    while(q_.size() >= capacity_) {
        producer_.wait(lock);
    }
    q_.push(std::move(item));
    consumer_.notify_one();
}

template<typename T>
bool BlockQueue<T>::Pop() {
    std::unique_lock<std::mutex> lock(mtx_);
    while(q_.empty()) {
        consumer_.wait(lock);
        if(closed_) {
            return false;
        }
    }
    q_.pop();
    producer_.notify_one();
    return true;
}

template<typename T>
void BlockQueue<T>::Flush() {
    consumer_.notify_one();
}