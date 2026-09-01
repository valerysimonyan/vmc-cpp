#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Make an object so we don't have to repeatedly reallocate memory when doing SR inverse, this will carry through every time we find the CG inverse
class ThreadPool {
    // Public facing part of class, constructor, destructor, and run which takes an int and returns nothing
    public:
        explicit ThreadPool(int n_workers);
        ~ThreadPool();

        void run(const std::function<void(int)>& fn);
        int n_workers() const {return n_workers_;}
        
    private: 
        void worker_loop(int id);

        // Declare variables here 
        int n_workers_;
        std::vector<std::thread> workers_;
        
        std::mutex mu_;
        std::condition_variable cv_work_;
        std::condition_variable cv_done_;
        
        const std::function<void(int)>* fn_ = nullptr;
        unsigned long generation_ = 0;
        int completed_ = 0;
        bool shutdown_ = false;

};