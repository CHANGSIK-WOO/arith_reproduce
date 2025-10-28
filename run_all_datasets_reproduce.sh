#!/bin/bash
#SBATCH --job-name=arith_reproduce
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=15G
#SBATCH --time=14-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail

# ============================================
# 모든 데이터셋 실행 (PACS, OfficeH, VLCS, TerraIncognita, DomainNet)
# ============================================

echo "========================================"
echo "Starting All Dataset Experiments"
echo "========================================"

# 1. PACS
echo ""
echo "========================================"
echo "[1/5] Running PACS experiments..."
echo "========================================"
bash shell/run_pacs_reproduce.sh

# 2. OfficeHome
echo ""
echo "========================================"
echo "[2/5] Running OfficeHome experiments..."
echo "========================================"
bash shell/run_officeh_reproduce.sh

# 3. VLCS
echo ""
echo "========================================"
echo "[3/5] Running VLCS experiments..."
echo "========================================"
bash shell/run_vlcs_reproduce.sh

# 4. TerraIncognita
echo ""
echo "========================================"
echo "[4/5] Running TerraIncognita experiments..."
echo "========================================"
bash shell/run_terra_reproduce.sh

# 5. DomainNet
echo ""
echo "========================================"
echo "[5/5] Running DomainNet experiments..."
echo "========================================"
bash shell/run_domainnet_reproduce.sh

echo ""
echo "========================================"
echo "All Dataset Experiments Completed!"
echo "========================================"


