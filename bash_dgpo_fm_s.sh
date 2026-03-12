#!/bin/bash

# 激活环境
source activate DGPO

# ==========================================
# 实验大盘配置 (Serial Benchmark Settings)
# ==========================================
gpu_id=0
timesteps=180000000               # 180M 步对齐原论文
#envs=("FingerSpin" "CheetahRun")  # 串行的话可以先少测几个环境
envs=("FingerSpin")  # 串行的话可以先少测几个环境
seeds=(1 2 3)                     # 3 个种子
alpha=0.1

mkdir -p logs

echo "🚀 [$(date '+%H:%M:%S')] 开始 DGPO 180M 步串行 (Serial) 对比实验..."
echo "⚠️  当前模式：排队执行，每次仅占用 1 个 GPU 进程。"

# 外层循环：遍历环境
for env in "${envs[@]}"; do
    echo "========================================"
    echo "🌟 正在攻克环境: $env"
    echo "========================================"

    # 内层循环：遍历随机种子
    for seed in "${seeds[@]}"; do
        
        # ------------------------------------------------
        # 1. 运行 DGPO (Flow Matching)
        # ------------------------------------------------
        time=$(date '+%Y-%m-%d-%H%M%S')
        log_fm="logs/dgpo_fm_${env}_${alpha}_seed${seed}_${time}.txt"
        
        echo "  ⏳ [正在运行] DGPO (FM) | Seed: $seed | 日志: $log_fm"
        XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train_dgpo_fm.py \
            --config.resampling_alpha $alpha \
            --env-name $env \
            --seed $seed \
            --config.num_timesteps $timesteps \
            > $log_fm 2>&1
            
        echo "  ✅ [运行完毕] DGPO (FM) | Seed: $seed"

    done
done

echo "========================================"
echo "🎉 所有实验全部串行执行完毕！"