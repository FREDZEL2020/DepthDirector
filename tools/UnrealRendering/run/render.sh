TASK=${1-"full_loop"}
echo TASK $TASK
source ~/miniconda3/etc/profile.d/conda.sh 

# 2. 激活环境
conda activate recam
# 3. 进入目录并执行
POD_NAME=$POD_NAME python run.py --config configs/ue_config/recam.json --storage configs/storage/local.json --task $TASK