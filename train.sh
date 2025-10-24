#!/bin/bash
#SBATCH --job-name=arith_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=10G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err
set -euo pipefail
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate arith

# GPU 설정
GPU_NUM=0
#export CUDA_VISIBLE_DEVICES=${GPU_NUM}

# Seed 리스트
SEEDS=(42)

# 공통 파라미터
SAVE_DIR="/data/changsik/arith/save"
EVAL_STEP=300

# ============================================
# PACS 실험
# ============================================
echo "Starting PACS experiments..."

DATASET="PACS"
BATCH_SIZE=16
NUM_EPOCH=6000
LR=2e-4
META_LR=1e-2
KNOWN_CLASSES="dog elephant giraffe horse guitar house person"

# PACS 모든 도메인
PACS_DOMAINS=("photo" "cartoon" "art_painting" "sketch")

for TARGET in "${PACS_DOMAINS[@]}"
do
    SOURCE_DOMAINS=()
    for DOMAIN in "${PACS_DOMAINS[@]}"
    do
        if [ "$DOMAIN" != "$TARGET" ]; then
            SOURCE_DOMAINS+=("$DOMAIN")
        fi
    done
    SOURCE_DOMAIN_STR="${SOURCE_DOMAINS[@]}"

    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="pacs_${TARGET}_seed${SEED}"

        echo "Running: ${SAVE_NAME}"

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

        echo "${DATASET} Completed: ${SAVE_NAME}"
        echo ""
    done
done

# ============================================
# VLCS 실험
# ============================================
#echo "Starting VLCS experiments..."
#
#DATASET="VLCS"
#BATCH_SIZE=16
#NUM_EPOCH=6000
#LR=2e-4
#META_LR=1e-2
#KNOWN_CLASSES="0 1 2 3 4"
#
#VLCS_DOMAINS=("CALTECH" "PASCAL" "SUN" "LABELME")
#
#for TARGET in "${VLCS_DOMAINS[@]}"
#do
#    SOURCE_DOMAINS=()
#    for DOMAIN in "${VLCS_DOMAINS[@]}"
#    do
#        if [ "$DOMAIN" != "$TARGET" ]; then
#            SOURCE_DOMAINS+=("$DOMAIN")
#        fi
#    done
#    SOURCE_DOMAIN_STR="${SOURCE_DOMAINS[@]}"
#
#    for SEED in "${SEEDS[@]}"
#    do
#        SAVE_NAME="vlcs_${TARGET}_seed${SEED}"
#
#        echo "Running: ${SAVE_NAME}"
#
#        python main.py \
#            --dataset ${DATASET} \
#            --source-domain ${SOURCE_DOMAIN_STR} \
#            --target-domain ${TARGET} \
#            --known-classes ${KNOWN_CLASSES} \
#            --gpu ${GPU} \
#            --seed ${SEED} \
#            --batch-size ${BATCH_SIZE} \
#            --num-epoch ${NUM_EPOCH} \
#            --eval-step ${EVAL_STEP} \
#            --lr ${LR} \
#            --meta-lr ${META_LR} \
#            --save-dir ${SAVE_DIR} \
#            --save-name ${SAVE_NAME} \
#            --save-later
#
#        echo "Completed: ${SAVE_NAME}"
#        echo ""
#    done
#done

echo "All experiments completed!"