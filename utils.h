#ifndef __KFM_UTILS_H__
#define __KFM_UTILS_H__
#include <string>
#include <fstream>
#include <Eigen/Eigen>

namespace KFM{

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

}
#endif
