from distutils.core import setup, Extension
import os
import sys

ret = os.system('make')
if ret != 0:
    echo "build protobuf failed"
    sys.exit(ret)

pybind11 = os.popen('python3 -m pybind11 --includes').read().strip().replace('-I', '').split(' ')
incs = []
incs.extend(pybind11)
incs.append('/usr/local/include/eigen3/')

kfm = Extension('KFM', include_dirs=incs, libraries=['protobuf'], sources=['FMRegressor.cpp', 'model.pb.cc'], language='c++')
setup(name='KFM', version='1.0', ext_modules=[kfm], include_dirs=incs)
