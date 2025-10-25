#!/bin/bash
#SBATCH --job-name=domainnet_full
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=15G
#SBATCH --time=7-0
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
DATASET="DomainNet"
BATCH_SIZE=16
NUM_EPOCH=6000  # DomainNet은 더 많은 epoch 필요
EVAL_STEP=3000
LR=2e-4
META_LR=1e-2

# DomainNet 클래스 (전체)
KNOWN_CLASSES="aircraft_carrier airplane alarm_clock ambulance angel animal_migration ant anvil apple arm asparagus axe backpack banana bandage barn baseball baseball_bat basket basketball bat bathtub beach bear beard bed bee belt bench bicycle binoculars bird birthday_cake blackberry blueberry book boomerang bottlecap bowtie bracelet brain bread bridge broccoli broom bucket bulldozer bus bush butterfly cactus cake calculator calendar camel camera camouflage campfire candle cannon canoe car carrot castle cat ceiling_fan cello cell_phone chair chandelier church circle clarinet clock cloud coffee_cup compass computer cookie cooler"

DOMAINNET_DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

# ============================================
# DomainNet 학습
# ============================================
echo "========================================"
echo "Starting DomainNet Training"
echo "========================================"

for TARGET in "${DOMAINNET_DOMAINS[@]}"
do
    # Source domain 생성 (target 제외)
    SOURCE_DOMAINS=()
    for DOMAIN in "${DOMAINNET_DOMAINS[@]}"
    do
        if [ "$DOMAIN" != "$TARGET" ]; then
            SOURCE_DOMAINS+=("$DOMAIN")
        fi
    done
    SOURCE_DOMAIN_STR="${SOURCE_DOMAINS[@]}"

    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="domainnet_${TARGET}_seed${SEED}"

        echo ""
        echo "========================================"
        echo "Training: ${SAVE_NAME}"
        echo "Source: ${SOURCE_DOMAIN_STR}"
        echo "Target: ${TARGET}"
        echo "========================================"

        python main.py \
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

        echo "✓ Training Completed: ${SAVE_NAME}"
        echo ""
    done
done

# ============================================
# DomainNet 평가
# ============================================
echo ""
echo "========================================"
echo "Starting DomainNet Evaluation"
echo "========================================"

for TARGET in "${DOMAINNET_DOMAINS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="domainnet_${TARGET}_seed${SEED}"

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

        echo "✓ Evaluation Completed: ${SAVE_NAME}"
        echo ""
    done
done

echo ""
echo "========================================"
echo "DomainNet All Experiments Completed!"
echo "========================================"