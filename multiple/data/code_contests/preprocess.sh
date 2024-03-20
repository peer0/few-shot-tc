#!/bin/bash 
 
echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 
 
# conda 환경 활성화. 
# source  ~/.bashrc 
cd /home/jungin/workspace/JointMatch/
source jointmatch/bin/activate
 
# cuda 11.0 환경 구성. 
export CUDA_VISIBLE_DEVICES=0

# ml unload cuda/11.2 nccl/2.8.4/cuda11.2 
# ml load cuda/11.0 nccl/2.8.4/cuda11.0 
# ml list 
 
cd /home/jungin/workspace/JointMatch/multiple/data/code_contests
 
# 활성화된 환경에서 코드 실행. 
# python -u problem_based_split.py
# python -u get_java_data.py
# python -u synonym_aug.py
python -u codegen.py
 
echo "###" 
echo "### END DATE=$(date)"