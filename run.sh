cd ~/hw6
# export CUDA_INSTALL_PATH=/opt/cuda
# source ~/gpgpu-sim_distribution/setup_environment

RunFile=./hw6-3

rm ${RunFile}.exe
nvcc ${RunFile}.cu -std=c++11 -O3 -o ${RunFile}.exe #-gencode arch=compute_61,code=compute_61 #-lcudart

rm gpgpu-sim.log
./${RunFile}.exe > gpgpu-sim.log
code gpgpu-sim.log