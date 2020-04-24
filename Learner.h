#ifndef __KFM_LEARNER_H__
#define __KFM_LEARNER_H__

#include <Eigen/Eigen>
#include <functional>
#include <iostream>
#include "utils.h"
#include <memory>
#include "ops.h"
#include "ParameterServer.h"
#include <thread>
#include <algorithm>

namespace KFM
{

class Learner
{
public:
    typedef std::function<void(Eigen::VectorXd const&, Eigen::VectorXd const&, Eigen::VectorXd&)> loss_grad_type;
    typedef std::function<double(Eigen::VectorXd const&, Eigen::VectorXd const&)> loss_op_type;
    typedef std::function<void(Eigen::VectorXd const&, Eigen::VectorXd&)> output_grad_type;
    typedef std::function<void(ModelPrivate const&, Eigen::MatrixXd const&, Eigen::VectorXd&, Eigen::MatrixXd&)> infer_op_type;
    virtual void fit(ParaemterServer& ps, int batch_size, int epoch=5, int nthreads=-1) = 0;
    
}; 


class SGDLearner : public Learner
{
public:

    SGDLearner(loss_grad_type const& loss_grad, output_grad_type const& output_grad, loss_op_type const& loss_op, infer_op_type const& infer_op, double lr=0.001, double lambda=0.001):
        _loss_grad(loss_grad),
        _output_grad(output_grad),
        _loss_op(loss_op),
        _infer_op(infer_op),
        _lr(lr)
    {

    }
    
    virtual void fit(ParaemterServer& ps, int batch_size, int epoch=5, int nthreads=-1) override
    {
        if (nthreads < 0){
            nthreads = std::thread::hardware_concurrency();
        }

        Eigen::Index rows;
        Eigen::Index cols;
        auto batch = batch_size;
        
        int gstep = 0;
        ps.get_data_shape(gstep, rows, cols);

        for(auto i = 0; i < epoch; ++i)
        {
            auto step = ps.step();
            auto max_step = ps.max_step();
            std::vector<std::thread> threads;
            for (auto j = 0; j < nthreads; ++j){
                threads.emplace_back(std::thread([batch, rows, nthreads, j, &ps, this, max_step](){
                    std::map<std::string, Eigen::MatrixXd> parameters;
                    int step = 0;
                    int ret = ps.get_parameters(step, parameters);
                    if (ret != 0){
                        return;
                    }

                    for (auto k = 0; k < rows/(batch*nthreads); ++k){
                        Eigen::Index start = nthreads*batch*k + j*batch; 
                        Eigen::Index end = start + batch;
                        start = std::min(start, rows);
                        end = std::min(end, rows);
                        if (start == end){
                            return;
                        }

                        Eigen::MatrixXd X;
                        Eigen::MatrixXd y;
                        auto s = k + step;
                        ret = ps.get_data(X, y, start, end);
                        if (ret != 0){
                            continue;
                        }

                        ModelPrivate p;
                        p.W = parameters["W"];
                        p.V = parameters["V"];
                        p.b = parameters["b"](0, 0);
                        Eigen::MatrixXd dW;
                        Eigen::MatrixXd dV;
                        double db;

                        auto loss = this->step(X, y, p, dV, dW, db);
                        std::cout << "loss = " << loss << std::endl;
                        Eigen::MatrixXd _db = Eigen::MatrixXd::Zero(1, 1);
                        _db(0, 0) = db;
                        std::map<std::string, Eigen::MatrixXd> dParameters = {
                            {"W", dW}, {"V", dV}, {"b", _db}
                        };
                        parameters["W"] += dW;
                        parameters["V"] += dV;
                        parameters["b"] += _db;

                        ret = ps.update_parameters(s, dParameters);
                        step += 1;
                        if (ret != 0){
                            continue;
                        }

                        if (s - step >= max_step){
                            ps.get_parameters(step, parameters);
                        }
                    }
                    
                }));
            }

            for (auto& t: threads){
                t.join();
            }

            std::cout << "epoch " << i << std::endl;
        }
    }

    virtual double step(Eigen::MatrixXd const& X, 
                        Eigen::VectorXd const& y, 
                        ModelPrivate const& paramters,
                        Eigen::MatrixXd& dV, 
                        Eigen::MatrixXd& dW, double& db) const 
    {
        auto const& V = paramters.V;
        Eigen::VectorXd g_loss;
        Eigen::VectorXd g_output;
        auto n = static_cast<double>(X.rows());

        Eigen::VectorXd yhat;
        Eigen::MatrixXd XV;
        _infer_op(paramters, X, yhat, XV);
        
        _loss_grad(y, yhat, g_loss);
        _output_grad(yhat, g_output);
        
        auto g_loss_output = g_loss.array() * g_output.array();
        db = -_lr*(g_loss_output.sum()/n) - _lr*_lambda*paramters.b;
        Eigen::Matrix<double, 1, Eigen::Dynamic> gt = g_loss_output.transpose();
        Eigen::MatrixXd Wl2 = -_lr*_lambda*paramters.W;
        dW = -_lr*((gt * X).array() / n);
        dW = dW + Wl2;
        
        Eigen::MatrixXd mask = g_loss_output.replicate(1, X.cols());
        Eigen::MatrixXd p1 = (X.array() * mask.array()).matrix().transpose();

        //Eigen::MatrixXd p2 = (Eigen::square(X.array()) * mask.array()).matrix();
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> p2 = Eigen::MatrixXd::Zero(V.rows(), V.cols()).array();
        for (Eigen::Index i = 0; i < X.rows(); ++i){
            auto row = X.row(i).transpose();
            auto t = V.array() * row.replicate(1, V.cols()).array() * gt(0, i);
            p2 = t + p2;
        }

        Eigen::MatrixXd Vl2 = -_lr*_lambda*paramters.V;
        dV = (-_lr/n) * (p1*XV -  p2.matrix());
        dV = dV + Vl2;

        return _loss_op(y, yhat);
    }

private:
    loss_grad_type _loss_grad;
    output_grad_type _output_grad;
    loss_op_type _loss_op;
    infer_op_type _infer_op;
    double _lr;
    double _lambda;
};

class LearnerFactory
{
public:
    LearnerFactory& operator=(LearnerFactory const&) = delete;
    LearnerFactory(LearnerFactory const&) = delete;

    static LearnerFactory& instance()
    {
        static LearnerFactory factory;
        return factory;
    }

    std::shared_ptr<Learner> create(LEARNER_t learner, OUTPUT_t output, double lr=0.001, double lambda=0.001)
    {
        auto infer_op = std::bind(_fm_infer, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, output);
        if (learner == SGD){
            if (output == LINER){
                return std::make_shared<SGDLearner>(mse_grad, liner_grad, mse_op, infer_op, lr, lambda);
            }else if (output == SIGMOID){
                return std::make_shared<SGDLearner>(logloss_grad, sigmoid_grad, logloss_op, infer_op, lr, lambda);
            }else{
                return nullptr;
            }
        }else{
            return nullptr;
        }
    }

private:
    LearnerFactory() {};
};

}

#endif
