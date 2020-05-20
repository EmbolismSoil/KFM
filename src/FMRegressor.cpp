#include "FMRegressor.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

static
KFM::FMRegressor create(std::string const& path)
{

}

namespace py=pybind11;

PYBIND11_MODULE(KFM, m)
{
    m.doc() = "FM Model";
    py::class_<KFM::FMRegressor>(m, "FMRegressor")
        .def(py::init<Eigen::Index, double, double, double, int, int>(), py::arg("ndim")=Eigen::Index(64), py::arg("lr")=double(0.0001), py::arg("gamma")=double(0.01), py::arg("eta")=double(0.1), py::arg("njobs")=1, py::arg("max_step")=int(1))
        .def("fit", &KFM::FMRegressor::fit, py::arg("X"), py::arg("y"), py::arg("batch_size")=int(100), py::arg("epoch")=int(5))
        .def("predict", &KFM::FMRegressor::predict)
        .def("save", &KFM::FMRegressor::save)
        .def("tostring", &KFM::FMRegressor::tostring)
        .def("W", &KFM::FMRegressor::W)
        .def("V", &KFM::FMRegressor::V)
        .def("b", &KFM::FMRegressor::b)
        .def("load", &KFM::FMRegressor::load);
}
