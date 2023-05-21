cd ~/hw6
export CUDA_VISIBLE_DEVICES=0

export CUDA_INSTALL_PATH=/opt/cuda
source ~/gpgpu-sim_distribution/setup_environment

RunFile=./hw6-3


rm *.exe
rm _*
rm *.ptx*
rm gpgpu-sim.log
rm gpgpu_inst_stats.txt
rm -r checkpoint_files

nvcc ${RunFile}.cu -o ${RunFile}.exe \
                   -gencode arch=compute_61,code=compute_61 \
                   -lcudart
                    #  --cudart shared # -lcudart

ldd ${RunFile}.exe # print out a list of dll

./${RunFile}.exe > gpgpu-sim.log
code gpgpu-sim.log




echo "--------------------------"
echo "$ ls -al"
echo ""
ls -al