#!/bin/bash
#SBATCH --job-name=terra_reproduce
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=15G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate arith

# ============================================
# 설정
# ============================================
GPU_NUM=0
SEEDS=(42)

SAVE_DIR="/data/changsik/arith/save"
DATASET="TerraIncognita"
BATCH_SIZE=12 # 32
NUM_EPOCH=2000 # 5000
EVAL_STEP=500
LR=5e-5
#LR=6e-6 # 5e-5
META_LR=1e-2

KNOWN_CLASSES="bobcat coyote dog opossum rabbit raccoon squirrel bird cat empty"
TERRA_DOMAINS=("location_38" "location_43" "location_46" "location_100")

# ============================================
# TerraIncognita 학습
# ============================================
echo "========================================"
echo "Starting TerraIncognita Training"
echo "========================================"

for TARGET in "${TERRA_DOMAINS[@]}"
do
    # Source domain 생성 (target 제외)
    SOURCE_DOMAINS=()
    for DOMAIN in "${TERRA_DOMAINS[@]}"
    do
        if [ "$DOMAIN" != "$TARGET" ]; then
            SOURCE_DOMAINS+=("$DOMAIN")
        fi
    done
    SOURCE_DOMAIN_STR="${SOURCE_DOMAINS[@]}"

    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="terra_${TARGET}_seed${SEED}_reproduce"

        echo ""
        echo "========================================"
        echo "Training: ${SAVE_NAME}"
        echo "Source: ${SOURCE_DOMAIN_STR}"
        echo "Target: ${TARGET}"
        echo "========================================"

        python main_reproduce.py \
            --dataset ${DATASET} \
            --source-domain ${SOURCE_DOMAIN_STR} \
            --target-domain ${TARGET} \
            --known-classes ${KNOWN_CLASSES} \
            --gpu ${GPU_NUM} \
            --seed ${SEED} \
            --batch-size ${BATCH_SIZE} \
            --num-epoch ${NUM_EPOCH} \
            --eval-step ${EVAL_STEP} \
            --lr ${LR} \
            --meta-lr ${META_LR} \
            --save-dir ${SAVE_DIR} \
            --save-name ${SAVE_NAME} \
            --save-later

        echo "✓ Training Completed: ${SAVE_NAME}_reproduce"
        echo ""
    done
done

# ============================================
# TerraIncognita 평가
# ============================================
echo ""
echo "========================================"
echo "Starting TerraIncognita Evaluation"
echo "========================================"

for TARGET in "${TERRA_DOMAINS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="terra_${TARGET}_seed${SEED}_reproduce"

        echo ""
        echo "========================================"
        echo "Evaluating: ${SAVE_NAME}"
        echo "Target: ${TARGET}"
        echo "========================================"

        python eval.py \
            --dataset ${DATASET} \
            --seed ${SEED} \
            --batch-size 128 \
            --hits 30 \
            --save-dir ${SAVE_DIR} \
            --save-name ${SAVE_NAME} \
            --gpu ${GPU_NUM}

        echo "✓ Evaluation Completed: ${SAVE_NAME}_reproduce"
        echo ""
    done
done

echo ""
echo "========================================"
echo "TerraIncognita All Experiments Completed!"
echo "========================================"


