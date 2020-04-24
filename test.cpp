#include <iostream>

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include "utils.h"
#include "Learner.h"
#include "FMModel.h"
#include <algorithm>
#include <cmath>
#include "ParameterServer.h"
#include "Learner.h"
#include <chrono>

using namespace Eigen;

int main(int argc, const char* argv[])
{
    if (argc != 5){
        std::cout << "usage: ./<proc> data.csv output.csv model.pb" << std::endl;
        return 0;
    }

    auto n = std::stoi(argv[4]);

    Eigen::MatrixXd m;
    KFM::load_csv(argv[1], m);
    Eigen::MatrixXd y = m.block(0, 0, m.rows(), 1).replicate(10, 1);
    Eigen::MatrixXd X = m.block(0, 1, m.rows(), m.cols()-1).replicate(10, 1);

    KFM::FMModel fm(13, 64, KFM::LINER);
    fm.randomInit();
    Eigen::MatrixXd W;
    Eigen::MatrixXd V;
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(1, 1);
    double _b;
    fm.getParameters(W, V, _b);
    b(0, 0) = _b;
    std::map<std::string, Eigen::MatrixXd> parameters = {
        {"W", W}, {"V", V}, {"b", b}
    };

    KFM::MultiThreadPS ps(X, y, parameters, 1); 
    auto learner = KFM::LearnerFactory::instance().create(KFM::SGD, KFM::LINER, 0.0001, 0.005);
    auto now = std::chrono::high_resolution_clock::now();
    learner->fit(ps, 200, 100, n);
    auto delta = std::chrono::high_resolution_clock::now() - now;
    std::cout << "fit cost = " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/1000 << "s" << std::endl;
    //学习到的参数设置到模型中
    int step = 0;
    ps.get_parameters(step, parameters);
    _b = parameters["b"](0, 0);
    W = parameters["W"];
    V = parameters["V"];
    fm.setParameters(W, V, _b);


    Eigen::VectorXd yhat;
    fm.predict(X, yhat);
    std::cout << "after " << step << " steps, loss = " << std::sqrt(KFM::mse_op(y, yhat)) << std::endl;
    std::ofstream ofs(argv[2]);
    
    ofs << "X:\n" << X << "y:\n" << y.transpose() << "\nyhat:\n" << yhat.transpose();
    fm.saveModel(argv[3]);

    auto fm1 = KFM::FMModel::loadModel(argv[3]);

    yhat = Eigen::MatrixXd::Zero(yhat.rows(), yhat.cols());
    fm1->predict(X, yhat);
    ofs << "\nyhat-restore:\n" << yhat.transpose();
    return 0;
}
