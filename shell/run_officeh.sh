#!/bin/bash
#SBATCH --job-name=officeh_full
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
DATASET="OfficeHome"
BATCH_SIZE=16
NUM_EPOCH=6000
EVAL_STEP=300
LR=2e-4
META_LR=1e-2

KNOWN_CLASSES="Alarm_Clock Backpack Batteries Bed Bike Bottle Bucket Calculator Calendar Candles Chair Clipboards Computer Couch Curtains Desk_Lamp Drill Eraser Exit_Sign Fan File_Cabinet Flipflops Flowers Folder Fork Glasses Hammer Helmet Kettle Keyboard Knives Lamp_Shade Laptop Marker Monitor Mop Mouse Mug Notebook Oven Pan Paper_Clip Pen Pencil Postit_Notes Printer Push_Pin Radio Refrigerator Ruler Scissors Screwdriver Shelf Sink Sneakers Soda Speaker Spoon TV Table Telephone ToothBrush Toys Trash_Can Webcam"

OFFICEH_DOMAINS=("Art" "Clipart" "Product" "RealWorld")

# ============================================
# OfficeHome 학습
# ============================================
echo "========================================"
echo "Starting OfficeHome Training"
echo "========================================"

for TARGET in "${OFFICEH_DOMAINS[@]}"
do
    # Source domain 생성 (target 제외)
    SOURCE_DOMAINS=()
    for DOMAIN in "${OFFICEH_DOMAINS[@]}"
    do
        if [ "$DOMAIN" != "$TARGET" ]; then
            SOURCE_DOMAINS+=("$DOMAIN")
        fi
    done
    SOURCE_DOMAIN_STR="${SOURCE_DOMAINS[@]}"

    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="officeh_${TARGET}_seed${SEED}"

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
# OfficeHome 평가
# ============================================
echo ""
echo "========================================"
echo "Starting OfficeHome Evaluation"
echo "========================================"

for TARGET in "${OFFICEH_DOMAINS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        SAVE_NAME="officeh_${TARGET}_seed${SEED}"

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
echo "OfficeHome All Experiments Completed!"
echo "========================================"


