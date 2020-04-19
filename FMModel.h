#ifndef __KFM_FM_MODEL_H__
#define __KFM_FM_MODEL_H__

#include <string>
#include <cstdint>
#include <Eigen/Eigen>
#include <string>
#include <iostream>
#include "ops.h"
#include "model.pb.h"
#include <fstream>
#include <memory>
#include "utils.h"

namespace KFM
{
template<typename LEARNER>
class FMModel
{
public:

    FMModel(Eigen::Index const nfeatures, Eigen::Index const ndim, OUTPUT_t const output, double lr=0.0001):
        _lr(lr)
    {
        if (output == SIGMOID){
            _learner = new LEARNER(logloss_grad, sigmoid_grad, logloss_op);
        }else{
            _learner = new LEARNER(mse_grad, liner_grad, mse_op);
        }
        _paramters.W = Eigen::MatrixXd::Zero(1, nfeatures);
        _paramters.V = Eigen::MatrixXd::Zero(nfeatures, ndim);
        _paramters.b = 0;
        _paramters.output = output;
    }

    static std::shared_ptr<FMModel> loadModel(std::string const& path)
    {
        ModelParameters model;
        std::ifstream ifs(path, std::ios::binary);
        
        if (!model.ParseFromIstream(&ifs)){
            return nullptr;
        }
        
        auto const& w = model.w();
        auto const& v = model.v();
        auto const b = model.b();
        double const lr = model.lr();
        auto const output = model.output();

        std::shared_ptr<FMModel> pfm (new FMModel(v.rows(), v.cols(), static_cast<KFM::OUTPUT_t>(output), lr));
        FMModel& fm = *pfm;
        //if (w.rows() != 1 || w.cols() != Eigen::Dynamic){
            //return nullptr;
        //}

        //if (v.rows() != Eigen::Dynamic || v.cols() != Eigen::Dynamic){
            //return nullptr;
        //}

        if (w.data_size() != w.rows() * w.cols() || v.data_size() != v.rows() * v.cols()){
            return nullptr;
        }

        for (auto i = 0; i < fm._paramters.W.rows(); i++){
            for(auto j = 0; j < fm._paramters.W.cols(); ++j){
                fm._paramters.W(i,j) = w.data(i*fm._paramters.W.cols() + j);
            }
        }

        for (auto i = 0; i < fm._paramters.V.rows(); ++i){
            for(auto j = 0; j < fm._paramters.V.cols(); ++j){
                fm._paramters.V(i, j) = v.data(i*fm._paramters.V.cols() + j);
            }
        }

        fm._paramters.b = b;
        fm._paramters.output = static_cast<OUTPUT_t>(output);
        fm._lr = lr;

        if (fm._paramters.output == SIGMOID){
            fm._learner = new LEARNER(logloss_grad, sigmoid_grad, logloss_op);
        }else{
            fm._learner = new LEARNER(mse_grad, liner_grad, mse_op);
        }

        return pfm;
    }

    int saveModel(std::string const& path)
    {
        ModelParameters model;
        auto w = model.mutable_w();
        for (auto i = 0; i < _paramters.W.rows(); ++i){
            for (auto j = 0; j < _paramters.W.cols(); ++j){
                w->add_data(_paramters.W(i, j));
            }
        }
        w->set_cols(static_cast<uint64_t>(_paramters.W.cols()));
        w->set_rows(static_cast<uint64_t>(_paramters.W.rows()));

        auto v = model.mutable_v();
        for (auto i = 0; i < _paramters.V.rows(); ++i){
            for(auto j = 0; j < _paramters.V.cols(); ++j){
                v->add_data(_paramters.V(i, j));
            }
        }

        v->set_cols(static_cast<uint64_t>(_paramters.V.cols()));
        v->set_rows(static_cast<uint64_t>(_paramters.V.rows()));

        model.set_b(_paramters.b);
        model.set_lr(_lr);
        model.set_output(static_cast<ModelParameters_OUTPUT>(_paramters.output));
        std::ofstream output(path, std::ios::binary);
        if (!model.SerializeToOstream(&output)){
            return -1;
        }

        return 0;
    }

    int predict(Eigen::MatrixXd const& X, Eigen::VectorXd& result) const
    {
        Eigen::MatrixXd XV;
        return _fm_infer(_paramters, X, result, XV);
    }

    int randomInit()
    {
        auto nfeatures = _paramters.V.rows();
        auto ndim = _paramters.V.cols();
        _paramters.V = Eigen::MatrixXd::Random(nfeatures, ndim);
        _paramters.W = Eigen::MatrixXd::Random(1, nfeatures);
        _paramters.b = Eigen::MatrixXd::Random(1, 1)(0, 0);
    }

    std::string const toString() const
    {
        std::stringstream ss;
        ss << "V:\n" << _paramters.V << "\nW:\n" << _paramters.W  << "\nb:\n" << _paramters.b;
        return ss.str();
    } 

    double fit(Eigen::MatrixXd const& X, Eigen::VectorXd const& y)
    {
        Eigen::MatrixXd XV;
        Eigen::VectorXd yhat;
        _fm_infer(_paramters, X, yhat, XV);

        Eigen::MatrixXd dV;
        Eigen::MatrixXd dW;
        double db;
        double loss = _learner->step(X, y, _paramters, dV, dW, db);
        _paramters.W -= dW * _lr;
        _paramters.V -= dV * _lr;
        _paramters.b -= db * _lr;

        return loss;
    }

    virtual ~FMModel()
    {
        delete _learner;
    }

private:



    FMModel() = default;

    ModelPrivate _paramters;

    LEARNER* _learner;
    double _lr;
};

};

#endif
