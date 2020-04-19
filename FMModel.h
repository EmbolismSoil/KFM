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

namespace KFM
{

template<typename LEARNER>
class FMModel
{
public:
    typedef enum{SIGMOID, LINER}OUTPUT_t;

    FMModel(OUTPUT_t const output, double lr=0.0001):
        _output(output),
        _lr(lr)
    {
        if (output == SIGMOID){
            _learner = new LEARNER(logloss_grad, sigmoid_grad, logloss_op);
        }else{
            _learner = new LEARNER(mse_grad, liner_grad, mse_op);
        }
    }

    static std::shared_ptr<FMModel> loadModel(std::string const& path)
    {
        std::shared_ptr<FMModel> pfm (new FMModel());
        FMModel& fm = *pfm;
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

        fm.randomInit(v.rows(), v.cols());
        //if (w.rows() != 1 || w.cols() != Eigen::Dynamic){
            //return nullptr;
        //}

        //if (v.rows() != Eigen::Dynamic || v.cols() != Eigen::Dynamic){
            //return nullptr;
        //}

        if (w.data_size() != w.rows() * w.cols() || v.data_size() != v.rows() * v.cols()){
            return nullptr;
        }

        for (auto i = 0; i < fm._W.rows(); i++){
            for(auto j = 0; j < fm._W.cols(); ++j){
                fm._W(i,j) = w.data(i*fm._W.cols() + j);
            }
        }

        for (auto i = 0; i < fm._V.rows(); ++i){
            for(auto j = 0; j < fm._V.cols(); ++j){
                fm._V(i, j) = v.data(i*fm._V.cols() + j);
            }
        }

        fm._b = b;
        fm._output = static_cast<OUTPUT_t>(output);
        fm._lr = lr;

        if (fm._output == SIGMOID){
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
        for (auto i = 0; i < _W.rows(); ++i){
            for (auto j = 0; j < _W.cols(); ++j){
                w->add_data(_W(i, j));
            }
        }
        w->set_cols(static_cast<uint64_t>(_W.cols()));
        w->set_rows(static_cast<uint64_t>(_W.rows()));

        auto v = model.mutable_v();
        for (auto i = 0; i < _V.rows(); ++i){
            for(auto j = 0; j < _V.cols(); ++j){
                v->add_data(_V(i, j));
            }
        }

        v->set_cols(static_cast<uint64_t>(_V.cols()));
        v->set_rows(static_cast<uint64_t>(_V.rows()));

        model.set_b(_b);
        model.set_lr(_lr);
        model.set_output(static_cast<KFM::ModelParameters_OUTPUT>(_output));
        std::ofstream output(path, std::ios::binary);
        if (!model.SerializeToOstream(&output)){
            return -1;
        }

        return 0;
    }

    int predict(Eigen::MatrixXd const& X, Eigen::VectorXd& result) const
    {
        Eigen::MatrixXd XV;
        return _infer(X, result, XV);
    }

    int randomInit(Eigen::Index const nfeatures, Eigen::Index const ndim)
    {
        _V = Eigen::MatrixXd::Random(nfeatures, ndim);
        _W = Eigen::MatrixXd::Random(1, nfeatures);
        _b = Eigen::MatrixXd::Random(1, 1)(0, 0);
    }

    std::string const toString() const
    {
        std::stringstream ss;
        ss << "V:\n" << _V << "\nW:\n" << _W  << "\nb:\n" << _b;
        return ss.str();
    } 

    double fit(Eigen::MatrixXd const& X, Eigen::VectorXd const& y)
    {
        Eigen::MatrixXd XV;
        Eigen::VectorXd yhat;
        _infer(X, yhat, XV);

        Eigen::MatrixXd dV;
        Eigen::MatrixXd dW;
        double db;
        double loss = _learner->step(X, y, XV, _V, yhat, dV, dW, db);
        _W -= dW * _lr;
        _V -= dV * _lr;
        _b -= db * _lr;

        return loss;
    }

    virtual ~FMModel()
    {
        delete _learner;
    }

private:

    int _infer(Eigen::MatrixXd const& X, Eigen::VectorXd& result, Eigen::MatrixXd& XV) const
    {
        XV = X*_V;
        auto b = Eigen::square(XV.array()).matrix();
        auto c = Eigen::square(X.array()).matrix() * Eigen::square(_V.array()).matrix();
        auto d = b - c;
        auto crossFeature = 0.5*d.rowwise().sum();
        if (_output == LINER){
            result = ((X*_W.transpose()  + crossFeature).array() + _b).matrix(); 
            //result = ((X*_W.transpose()).array() + _b).matrix(); 
        }else{
            sigmoid_op(((X*_W.transpose() + crossFeature).array() + _b).matrix(), result); 
            //sigmoid_op(((X*_W.transpose()).array() + _b).matrix(), result); 
        }
        return 0;
    }

    FMModel() = default;

    Eigen::MatrixXd _V;
    Eigen::MatrixXd _W;

    double _b;
    OUTPUT_t _output;
    LEARNER* _learner;
    double _lr;
};

};

#endif
