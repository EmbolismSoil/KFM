#ifndef __KFM_LEARNER_H__
#define __KFM_LEARNER_H__

#include <Eigen/Eigen>
#include <functional>
#include <iostream>

namespace KFM
{

class SGDLearner
{
public:
    typedef std::function<void(Eigen::VectorXd const&, Eigen::VectorXd const&, Eigen::VectorXd&)> loss_grad_type;
    typedef std::function<double(Eigen::VectorXd const&, Eigen::VectorXd const&)> loss_op_type;
    typedef std::function<void(Eigen::VectorXd const&, Eigen::VectorXd&)> output_grad_type;

    SGDLearner(loss_grad_type const& loss_grad, output_grad_type const& output_grad, loss_op_type const& loss_op):
        _loss_grad(loss_grad),
        _output_grad(output_grad),
        _loss_op(loss_op)
    {

    }

    virtual double step(Eigen::MatrixXd const& X, 
                        Eigen::VectorXd const& y, 
                        Eigen::MatrixXd const& XV,
                        Eigen::MatrixXd const& V,
                        Eigen::VectorXd const& yhat, 
                        Eigen::MatrixXd& dV, 
                        Eigen::MatrixXd& dW, double& db)
    {
        
        Eigen::VectorXd g_loss;
        Eigen::VectorXd g_output;
        auto n = static_cast<double>(X.rows());
        
        _loss_grad(y, yhat, g_loss);
        _output_grad(yhat, g_output);
        
        auto g_loss_output = g_loss.array() * g_output.array();
        db = (g_loss_output.sum()/n);
        Eigen::Matrix<double, 1, Eigen::Dynamic> gt = g_loss_output.transpose();
        dW = ((gt * X).array() / n);
        
        Eigen::MatrixXd mask = g_loss_output.replicate(1, X.cols());
        Eigen::MatrixXd p1 = (X.array() * mask.array()).matrix().transpose();

        //Eigen::MatrixXd p2 = (Eigen::square(X.array()) * mask.array()).matrix();
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> p2 = Eigen::MatrixXd::Zero(V.rows(), V.cols()).array();
        for (Eigen::Index i = 0; i < X.rows(); ++i){
            auto row = X.row(i).transpose();
            auto t = V.array() * row.replicate(1, V.cols()).array() * gt(0, i);
            p2 = t + p2;
        }

        dV = (1.0/n) * (p1*XV -  p2.matrix());

        return _loss_op(y, yhat);
    }

private:
    loss_grad_type _loss_grad;
    output_grad_type _output_grad;
    loss_op_type _loss_op;
};

}

#endif
