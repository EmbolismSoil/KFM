#ifndef __KFM_OPS_H__
#define __KFM_OPS_H__

#include <Eigen/Eigen>

namespace KFM
{

    void sigmoid_op(Eigen::VectorXd const& x, Eigen::VectorXd& y)
    {
        y = (1.0 + (-x).array().exp()).inverse().matrix();
    }

    void sigmoid_grad(Eigen::VectorXd const& x, Eigen::VectorXd& grad)
    {
        grad = (x.array() * (1 - x.array())).matrix();
    }

    double logloss_op(Eigen::VectorXd const& y, Eigen::VectorXd const& yhat)
    {
        auto y_arr = y.array();
        auto yhat_arr = yhat.array();
        auto loss = y_arr*Eigen::log(yhat_arr) + (1 - y_arr)*Eigen::log(1 - yhat_arr);
        return loss.sum();
    }

    void logloss_grad(Eigen::VectorXd const& y, Eigen::VectorXd const& yhat, Eigen::VectorXd& grad)
    {
        auto y_arr = y.array();
        auto yhat_arr = yhat.array();

        grad = (y_arr/yhat_arr - (y_arr - 1)/(1 - yhat_arr)).matrix();
    }

    double mse_op(Eigen::VectorXd const& y, Eigen::VectorXd const& yhat)
    {
        auto yarr = y.array();
        auto yhat_arr = yhat.array();
        return Eigen::square(yarr - yhat_arr).mean();
    }

    void mse_grad(Eigen::VectorXd const& y, Eigen::VectorXd const& yhat, Eigen::VectorXd& grad)
    {
        grad = (2.0*(yhat.array() - y.array())).matrix();
    }

    void liner_grad(Eigen::VectorXd const& x, Eigen::VectorXd& grad)
    {
        grad = Eigen::MatrixXd::Ones(x.rows(), x.cols());
    }
};

#endif
