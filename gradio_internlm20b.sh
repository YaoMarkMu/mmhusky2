#!/usr/bin/env bash
set -x
T=$(date +%Y%m%d_%H%M%S)

work_dir=/mnt/petrelfs/zhangqinglong/Documents/Husky

env='/mnt/petrelfs/zhangqinglong/anaconda3/envs/zhangql/bin'
export PYTHONPATH=${work_dir}:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/mnt/petrelfs/zhangqinglong/.cache/torch_extensions
export PATH=/mnt/petrelfs/share/gcc/gcc-7.5.0/bin:$PATH
export PATH=/mnt/petrelfs/share_data/llm_env/dep/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share_data/llm_env/dep/cuda-11.7/lib64:$LD_LIBRARY_PATH

# 进入工作目录
cd $work_dir
srun -p INTERN2 --job-name='husky_multi_test' --gres=gpu:1 --cpus-per-task=8 --quotatype="auto" \
  ${env}/python -u ${work_dir}/gradio/mmhusky2/gradio_internlm20b.py \
  --port 10054 \
  --model_path /mnt/petrelfs/share_data/zhangqinglong/Husky/work_dirs/multi-modal/mmhusky_v2_21b_stage4_v1.0_fp16 \
  --gradio_head 1101_mmhusky_v2_21b_stage4_v1.0_fp16 \
  --model_type internlm20b \
  --use_root_path \
  --root_path mmhusky_v2_21b_stage4_v1
