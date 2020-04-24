#ifndef __THREADPOOL_HPP__
#define __THREADPOOL_HPP__

#include <thread>
#include <atomic>
#include <list>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

namespace KLib
{
class ThreadPool
{
public:
    ThreadPool(int threadNum):
        _threadNum(threadNum),
        _done(false)
    {
        if (_threadNum <= 0)
        {
            _threadNum = 1;
        }
    }

    void init()
    {
        for (int cnt = 0; cnt < _threadNum; ++cnt)
        {
            std::shared_ptr<std::thread> t(new std::thread(std::bind(&ThreadPool::_run, this)));
            _threads.push_back(t);
        }
    }

    std::future<int> post(std::function<int(void)> const& task)
    {
        std::packaged_task<int(void)> pk(task);
        auto res = pk.get_future();
        {
            std::lock_guard<std::mutex> guard(_mtx);
            _tasks.emplace_back(std::move(pk));
        }
        
        //惊群
        _cond.notify_all();
        return res;
    }

    virtual ~ThreadPool()
    {
        _done = true;
        _cond.notify_all();
        for (std::vector<std::shared_ptr<std::thread> >::const_iterator pos = _threads.begin(); pos != _threads.end(); ++pos)
        {
            (*pos)->join();
        }
    }

private:
    void _run()
    {
        for (;;){
            std::unique_lock<std::mutex> lk(_mtx);
            _cond.wait(lk, std::bind(&ThreadPool::_hasTask, this));
            if (_done){
                break; //退出
            }

            std::packaged_task<int(void)> task(std::move(_tasks.back()));
            _tasks.pop_back();
            lk.unlock();
    
            task();
        }
    }

    bool _hasTask()
    {
        return !_tasks.empty() || _done;
    }

    int  _threadNum;
    std::atomic<bool> _done;
    std::list<std::packaged_task<int(void)> > _tasks;
    std::vector<std::shared_ptr<std::thread> > _threads;
    std::mutex _mtx;
    std::condition_variable _cond;
};
}

#endif
