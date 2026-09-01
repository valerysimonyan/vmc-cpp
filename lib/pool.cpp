#include "pool.h"

// Upon creating thread pool reserve that number of thread, pass function with argument id, this assigns it to this instance of the class 
ThreadPool::ThreadPool(int n_workers) : n_workers_(n_workers) {
    workers_.reserve(n_workers_);
    for (int id = 0; id < n_workers_; id++) {
        workers_.emplace_back([this, id]() {worker_loop(id);});
    }
}

// Destroy thread pool, runs automatically when process end. First make data inaccesible to be read, then pause workers
ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        shutdown_ = true;
        generation_++;
    }
    cv_work_.notify_all();
    for (auto& w: workers_) w.join();
}

// Have each thread work on running function, when all done notify of completion. 
void ThreadPool::worker_loop(int id) {
    unsigned long seen_generation = 0;

    while (true) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_work_.wait(lock, [&]() {return generation_ != seen_generation;});
        seen_generation = generation_;
        if (shutdown_) return;
        const std::function<void(int)>* fn = fn_;
        lock.unlock();

        (*fn)(id);
        
        lock.lock();
        completed_++;
        if (completed_ == n_workers_) cv_done_.notify_one();
    }
}

// Manages data availability
void ThreadPool::run(const std::function<void(int)>& fn) {
    std::unique_lock<std::mutex> lock(mu_);
    fn_ = &fn; 

    completed_ = 0;
    generation_++;
    lock.unlock();
    cv_work_.notify_all();

    lock.lock();
    cv_done_.wait(lock, [this]() {return completed_ == n_workers_;});
}