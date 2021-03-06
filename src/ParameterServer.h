#ifndef __KFM_PARAMETERSERVER_H__
#define __KFM_PARAMETERSERVER_H__

#include <map>
#include <string>
#include <Eigen/Eigen>
#include <set>
#include <map>
#include <pthread.h>
#include "RWLock.h"

namespace  KFM
{

class ParaemterServer
{
public:
    //输入参数step: 当前工作节点的step，返回step为服务器step
    //输入参数node: 当前工作节点的唯一id
    virtual int get_parameters(int& step, std::map<std::string, Eigen::MatrixXd>& parameters) const = 0;
    virtual int update_parameters(int& step, std::map<std::string, Eigen::MatrixXd> const& parameters) = 0;
    virtual int dc_update_parameter(int& step, std::map<std::string, Eigen::MatrixXd> const& grad, std::map<std::string, Eigen::MatrixXd> const& param) = 0;
    virtual int get_data(Eigen::MatrixXd& X, Eigen::MatrixXd& y, int start, int end) const = 0;
    virtual int get_data_shape(int &step, Eigen::Index& rows, Eigen::Index& cols) const = 0;
    virtual int start() = 0;
    virtual int max_step() const = 0;
    virtual int step() const = 0; 
    virtual ~ParaemterServer() = default;
};

class MultiThreadPS: public ParaemterServer
{
public:
    MultiThreadPS(Eigen::MatrixXd const& X, Eigen::MatrixXd const& y, std::map<std::string, Eigen::MatrixXd> const& parameters, double lambda, double lr, int const max_step=0):
        _parameters(parameters),
        _lambda(lambda),
        _lr(lr),
        _max_step(max_step)
    {
        _X = X;
        _y = y;
        _step = 0;
    }

    MultiThreadPS(MultiThreadPS const& ) = delete;
    MultiThreadPS& operator=(MultiThreadPS const&) = delete;

    virtual ~MultiThreadPS(){
    }

    virtual int get_parameters(int& step, std::map<std::string, Eigen::MatrixXd>& parameters) const override
    {
        RLockGuard guard(_mtx);
        step = _step;
        parameters = _parameters;
        return 0;
    }

    virtual int update_parameters(int& step, std::map<std::string, Eigen::MatrixXd> const& parameters) override
    {
        WLockGaurd guard(_mtx);
        
        if (step < _step - _max_step){
            return -1;
        }

        for (auto const& item: parameters){
            if (_parameters.find(item.first) == _parameters.cend()){
                continue;
            }
            auto& p = _parameters[item.first];
            if (p.rows() != item.second.rows() || p.cols() != item.second.cols()){
                continue;
            }

            _parameters[item.first] -= _lr*item.second;
        }

        _step += 1;
        step = _step;
        return 0;
    }

    virtual int dc_update_parameter(int& step, std::map<std::string, Eigen::MatrixXd> const& grad, std::map<std::string, Eigen::MatrixXd> const& param) override
    {
        WLockGaurd guard(_mtx);
        
        if (step < _step - _max_step || grad.size() != param.size()){
            return -1;
        }

        for (auto const& p: param){
            auto ppos = _parameters.end();
            auto gpos = grad.cend();
            if ((ppos = _parameters.find(p.first)) == _parameters.end() || 
                                    (gpos = grad.find(p.first)) == grad.cend())
            {
                return -1;
            }

            auto const& w_t = p.second;
            auto const& w_tau = ppos->second;
            auto const& g_t = gpos->second;

            Eigen::MatrixXd dW = g_t + (_lambda*g_t.array()*g_t.array()*(w_tau - w_t).array()).matrix();
            ppos->second = w_tau - _lr*dW;

        }
        
        _step += 1;
        step = _step;
        return 0;
    }

    virtual int get_data(Eigen::MatrixXd& X, Eigen::MatrixXd& y, int start = -1, int end = -1) const override
    {
        if (start < 0 && end < 0){
            X = _X;
            y = _y;
        }else if (start*end < 0){
            return -1;
        }else if (start >= end){
           return -1;
        }else if (end >= _X.rows()){
            return -1;
        }else{
            X = _X.block(start, 0, _X.rows() - start, _X.cols());
            y = _y.block(start, 0, _X.rows() - start, _y.cols());
        }

        return 0;
    }

    virtual int get_data_shape(int& step, Eigen::Index& rows, Eigen::Index& cols) const override
    {
        step = _step;
        rows = _X.rows();    
        cols = _X.cols();
        return 0;
    }

    virtual int start() override
    {
        _step = 0;
        return 0;
    }

    virtual int step() const override
    {
        return _step;
    }

    virtual int max_step() const override
    {
        return _max_step;
    }

private:
    mutable RWMutex _mtx;
    int _step;
    Eigen::MatrixXd _X;
    Eigen::MatrixXd _y;
    std::map<std::string, Eigen::MatrixXd> _parameters;
    double _lambda;
    double _lr;
    int _max_step;
};

}
#endif
