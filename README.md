# RoboticAI2023_TVM
## Introduction
This project introduces methods for optimizing convolution using TVM deep learning compiler.
## Installation
### Install TVM
1. Clone the repository
```
git clone --recursive https://github.com/apache/tvm tvm
git submodule init
git submodule update
```
2. Conda environment setup (in cloned tvm/ directory)
```
# Create a conda environment with the dependencies specified by the yaml
conda env create --file conda/build-environment.yaml
# Activate the created environment
conda activate tvm-build
```
3. Create build directory, and edit the cmake file
```
mkdir build
cp cmake/config.cmake build
```
Edit build/config.cmake to customize the compilation options
- Change set(USE_CUDA OFF) to set(USE_CUDA ON) to enable CUDA backend.
- Change set(USE_LLVM OFF) to set(USE_LLVM ON).
4. Build TVM
```
cd build
cmake ..
make -j4
```
5. Python setup
Set the environment variable PYTHONPATH to tell python where to find the library. For example, assume we cloned tvm on the directory /path/to/tvm then we can add the following line in ~/.bashrc. The changes will be immediately reflected once you pull the code and rebuild the project (no need to call setup again)
```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
```
conda install numpy decorator attrs
conda install -c conda-forge gcc=11.4.0
conda install -c conda-forge gxx
# install pytorch with cuda (https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Check the tvm successfully installed
```
import tvm
```
(If there is any problem during installation, see my conda environment in conda/environment.yml or please let me know)
(Reference: https://tvm.apache.org/docs/install/from_source.html)
### Clone this project
```
git clone https://github.com/WoongQ/RoboticAI2023_TVM.git
```
## Run the codes
1. Hand-optimized convolution
```
python 1_opt_conv_manual.py
```
2. Using AutoTVM
```
python 2_opt_conv_autotvm.py
```
3. Using AutoScheduler
```
python 3_opt_conv_autoscheduler.py
```
