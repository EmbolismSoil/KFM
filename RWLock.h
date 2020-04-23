#ifndef __KFM_RWLOCK_H__
#define __KFM_RWLOCK_H__

#include <pthread.h>

namespace KFM
{

class RWMutex
{
public:
    RWMutex(){
        ::pthread_rwlock_init(&_lock, nullptr);
    }

    virtual ~RWMutex(){
        ::pthread_rwlock_destroy(&_lock);
    }

    void rdlock()
    {
        ::pthread_rwlock_rdlock(&_lock);
    }

    void wrlock()
    {
        ::pthread_rwlock_wrlock(&_lock);
    }

    void unlock()
    {
        ::pthread_rwlock_unlock(&_lock);
    }

private:
    ::pthread_rwlock_t _lock;
};

class RLockGuard
{
public:
    RLockGuard(RWMutex& lock):
        _mtx(lock)
    {
        _mtx.rdlock();
    }

    virtual ~RLockGuard()
    {
        _mtx.unlock();
    }
private:
    RWMutex& _mtx;    
};

class WLockGaurd
{
public:
    WLockGaurd(RWMutex& lock):
        _mtx(lock)
    {
        _mtx.wrlock();
    }

    virtual ~WLockGaurd(){
        _mtx.unlock();
    }
private:
    RWMutex& _mtx;
};

}



#endif
