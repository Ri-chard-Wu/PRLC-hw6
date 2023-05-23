# cd ~/hw6
# export CUDA_INSTALL_PATH=/opt/cuda
# source ~/gpgpu-sim_distribution/setup_environment
# nvcc test.cu -o test.exe -lcudart -gencode arch=compute_61,code=compute_61
# ./test.exe > test-sim.log
# code test-sim.log


# ------------------------------------


cd ~/hw6
export CUDA_VISIBLE_DEVICES=0

export CUDA_INSTALL_PATH=/opt/cuda
source ~/gpgpu-sim_distribution/setup_environment


# RunFile=./test
RunFile=./hw6-3

rm *.exe
rm _*
rm *.ptx*
# rm gpgpu-sim.log
rm gpgpu_inst_stats.txt
rm -r checkpoint_files

nvcc ${RunFile}.cu -o ${RunFile}.exe \
                    -gencode arch=compute_61,code=compute_61 \
                    -lcudart 
                   
                   

# ldd ${RunFile}.exe # print out a list of dll


./${RunFile}.exe > ${RunFile}-sim.log
python3 extract_stat.py ${RunFile}-sim.log ${RunFile}-sim.stat

# code ${RunFile}-sim.log
code ${RunFile}-sim.stat



# echo "--------------------------"
# echo "$ ls -al"
# echo ""
# ls -al