#ifndef __KFM_UTILS_H__
#define __KFM_UTILS_H__
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include "ops.h"

namespace KFM{
typedef enum{SIGMOID=0, LINER}OUTPUT_t;
typedef enum{SGD=0}LEARNER_t;
struct ModelPrivate
{
    Eigen::MatrixXd W;
    Eigen::MatrixXd V;
    double b;
}; 

static int _fm_infer(ModelPrivate const& paramters, Eigen::MatrixXd const& X, Eigen::VectorXd& result, Eigen::MatrixXd& XV, OUTPUT_t output) 
{
    auto const& V = paramters.V;
    auto const& W = paramters.W;
    auto const _b = paramters.b;

    XV = X*V;
    auto b = Eigen::square(XV.array()).matrix();
    auto c = Eigen::square(X.array()).matrix() * Eigen::square(V.array()).matrix();
    auto d = b - c;
    auto crossFeature = 0.5*d.rowwise().sum();
    if (output == LINER){
        result = ((X*W.transpose()  + crossFeature).array() + _b).matrix(); 
        //result = ((X*_W.transpose()).array() + _b).matrix(); 
    }else{
        sigmoid_op(((X*W.transpose() + crossFeature).array() + _b).matrix(), result); 
        //sigmoid_op(((X*_W.transpose()).array() + _b).matrix(), result); 
    }
    return 0;
}

template<typename M>
void load_csv (const std::string & path, M& m) 
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));

        }
        ++rows;

    }
    m = Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}    

template<typename M>
void vector2Matrix(M& m, std::vector<typename M::Scalar> const& vec, Eigen::Index rows)
{
    m = Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(vec.data(), rows, vec.size()/rows);
}

}
#endif
