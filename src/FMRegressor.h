#ifndef __KFM_FMREGRESSOR_H__
#define __KFM_FMREGRESSOR_H__

#include "FMModel.h"
namespace KFM
{

class FMRegressor
{
public:
    FMRegressor(Eigen::Index const ndim=64, double lr=0.0001, double gamma=0.01, double lambda=0.1, int njobs=-1, int max_step=-1):
        _ndim(ndim),
        _fit(false),
        _lambda(lambda),
        _lr(lr),
        _gamma(gamma),
        _max_step(max_step),
        _njobs(njobs)
    {
        _njobs = _njobs > 0 ? _njobs : std::thread::hardware_concurrency();
        _max_step = _max_step > 0 ? _max_step : _njobs;
    }

    FMRegressor& fit(Eigen::MatrixXd const& X, Eigen::MatrixXd const& y, int const batch_size=100, int const epoch=5)
    {
        _model = std::make_shared<FMModel>(X.cols(), _ndim, LINER);
        _model->randomInit();
        Eigen::MatrixXd W;
        Eigen::MatrixXd V;
        Eigen::MatrixXd b = Eigen::MatrixXd::Zero(1, 1);
        double _b;
        _model->getParameters(W, V, _b);
        b(0, 0) = _b;

        std::map<std::string, Eigen::MatrixXd> parameters = {
            {"W", W}, {"V", V}, {"b", b}
        };

        KFM::MultiThreadPS ps(X, y, parameters, _lambda, _lr, _max_step);
        auto learner = LearnerFactory::instance().create(KFM::SGD, KFM::LINER, _gamma, _njobs);
        learner->fit(ps, batch_size, epoch);
        
        int step = 0;
        ps.get_parameters(step, parameters);
        _b = parameters["b"](0, 0);
        W = parameters["W"];
        V = parameters["V"];
        _model->setParameters(W, V, _b);
        _fit = true;
        return *this;
    }

    void save(std::string const& path)
    {
        if (!_fit)
        {
            throw std::runtime_error("assert(fit == true) failed.");
        }
        _model->saveModel(path);
    } 

    int load(std::string const& path)
    {
        _model = FMModel::loadModel(path);
        _fit = true;
    }

    //Eigen::VectorXd predict(Eigen::Ref<Eigen::MatrixXd> const X)
    Eigen::VectorXd predict(Eigen::MatrixXd const& X)
    {
        if (!_fit){
            throw std::runtime_error("assert(fit == true) failed.");
        }
        Eigen::VectorXd y;
        _model->predict(X, y);
        return y;
    }

    std::string tostring()
    {
        std::stringstream ss;
        Eigen::MatrixXd W;
        Eigen::MatrixXd V;
        double b;
        _model->getParameters(W, V, b);

        ss << "bias: " << b << "\nW:\n" << W << "\nV:\n" << V;
        return ss.str();
    }

    Eigen::MatrixXd W() const
    {
        Eigen::MatrixXd W;
        Eigen::MatrixXd V;
        double b;
        _model->getParameters(W, V, b);
        return W;
    }

    Eigen::MatrixXd V() const
    {
        Eigen::MatrixXd W;
        Eigen::MatrixXd V;
        double b;
        _model->getParameters(W, V, b);
        return V;
    }

    double b() const
    {
        Eigen::MatrixXd W;
        Eigen::MatrixXd V;
        double b;
        _model->getParameters(W, V, b);
        return b;
    }

private:
    std::shared_ptr<FMModel> _model;    
    Eigen::Index _ndim;
    bool _fit;
    double _lambda;
    double _lr;
    double _gamma;
    int _max_step;
    int _njobs;
};

}


#endif
