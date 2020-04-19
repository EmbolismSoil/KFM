#include <iostream>

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include "utils.h"
#include "Learner.h"
#include "FMModel.h"
#include <algorithm>
#include <cmath>

using namespace Eigen;

int main(int argc, const char* argv[])
{
    if (argc != 4){
        std::cout << "usage: ./<proc> data.csv output.csv model.pb" << std::endl;
        return 0;
    }

    Eigen::MatrixXd m;
    KFM::load_csv(argv[1], m);
    auto y = m.block(0, 0, m.rows(), 1);
    auto X = m.block(0, 1, m.rows(), m.cols()-1);

    KFM::FMModel<KFM::SGDLearner> fm(13, 64,KFM::LINER, 0.0001);
    fm.randomInit();

    std::cout << "X.rows() = " << X.rows() << "\nX.cols() = " << X.cols() << std::endl;
    std::cout << "FM:\n" << fm.toString() << std::endl;
    for (auto j = 0; j < 100; ++j){
        for (auto i = 0; i < X.rows() / 50; ++i){
            int start = i*50;
            int size = std::min(50, static_cast<int>(X.rows() - start));
            auto loss = fm.fit(X.block(start, 0, size, X.cols()), y.block(start, 0, size, 1));
            std::cout << "step: " << j*X.rows()/5 + i << ", loss: " << std::sqrt(loss) << std::endl;
        }
    }

    Eigen::VectorXd yhat;
    fm.predict(X, yhat);
    std::ofstream ofs(argv[2]);
    
    ofs << "X:\n" << X << "y:\n" << y.transpose() << "\nyhat:\n" << yhat.transpose();
    fm.saveModel(argv[3]);

    auto fm1 = KFM::FMModel<KFM::SGDLearner>::loadModel(argv[3]);

    yhat = Eigen::MatrixXd::Zero(yhat.rows(), yhat.cols());
    fm1->predict(X, yhat);
    ofs << "\nyhat-restore:\n" << yhat.transpose();
    return 0;
}
