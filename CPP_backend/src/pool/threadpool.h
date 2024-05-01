#pragma once
#include <atomic>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <functional>
#include <utility>

class ThreadPool {
public:
    // Constructor
    ThreadPool(size_t thread_num = std::thread::hardware_concurrency());

    // Destructor
    ~ThreadPool(); 

    // Add a task to the thread pool
    template<class F>
    void AddTask(F&& task);

private:
    // Function to run by each thread in the pool
    void Running();

    // Vector to store worker threads
    std::vector<std::thread> threads_;

    // Queue to store tasks
    std::queue<std::function<void()>> tasks_;      

    // Mutex for thread safety
    std::mutex mtx_;                    

    // Condition variable for synchronization           
    std::condition_variable cond_;                 

    // Flag indicating whether the pool is closed
    std::atomic<bool> is_closed_;                  
};

template<class F>
void ThreadPool::AddTask(F&& task) {
    if(is_closed_) throw std::logic_error("Cannot add task to a closed ThreadPool.");
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.emplace(std::forward<F>(task));
    cond_.notify_one();
}




