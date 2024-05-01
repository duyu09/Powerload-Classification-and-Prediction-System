#pragma once

#include <set>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <stdexcept>
#include "../log/log.h"

using Callback = std::function<void()>;
using Clock = std::chrono::steady_clock;
using Duration = std::chrono::milliseconds;

class Timer {
public:
    Timer() = default;
    ~Timer() = default;

    void Adjust(size_t id, size_t timeout) {
        if (!Contains(id)) {
            throw std::out_of_range("Timer ID not found.");
        }
        auto& node = callbacks_.at(id);
        node.expires = Clock::now() + Duration(timeout);
        UpdateHeap(node);
    }

    void Add(size_t id, size_t timeout, Callback cb) {
        Node node{id, Clock::now() + Duration(timeout), cb};
        InsertNode(node);
    }

    void Pop() {
        if (heap_.empty()) {
            return;
        }
        auto it = heap_.begin();
        callbacks_.erase(it->id);
        heap_.erase(it);
    }

    void Clear() {
        while (!heap_.empty()) {
            Pop();
        }
    }

    void Work(size_t id) {
        if (!Contains(id)) {
            throw std::out_of_range("Timer ID not found.");
        }
        auto& node = callbacks_.at(id);
        node.cb();
        heap_.erase(node);
        callbacks_.erase(id);
    }

    void Tick() {
        while (!heap_.empty()) {
            auto now = Clock::now();
            auto& node = *heap_.begin();
            if (node.expires > now) {
                break;
            }
            node.cb();
            Pop();
        }
    }

    int GetNextTick() {
        Tick();
        if (heap_.empty()) {
            return -1;
        }
        auto now = Clock::now();
        int next_tick = std::chrono::duration_cast<Duration>(heap_.begin()->expires - now).count();
        return next_tick > 0 ? next_tick : 0;
    }

private:
    struct Node {
        size_t id;
        Clock::time_point expires;
        Callback cb;
        bool operator<(const Node& other) const {
            return expires < other.expires;
        }
    };

    bool Contains(size_t id) const {
        return callbacks_.count(id) > 0;
    }

    void InsertNode(const Node& node) {
        heap_.insert(node);
        callbacks_[node.id] = node;
    }

    void UpdateHeap(Node& node) {
        heap_.erase(node);
        heap_.insert(node);
    }

    std::unordered_map<size_t, Node> callbacks_;
    std::set<Node> heap_;
};
