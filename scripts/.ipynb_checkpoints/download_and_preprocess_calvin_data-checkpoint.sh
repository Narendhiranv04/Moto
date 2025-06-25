#!/usr/bin/env bash
# ------------------------------------------------------------
#  download_and_preprocess_calvin_split.sh
#
#  Usage:
#     # minimal: uses defaults shown below
#     bash download_and_preprocess_calvin_split.sh
#
#     # override the split (e.g. task_C_B)
#     bash download_and_preprocess_calvin_split.sh task_C_B
#
#  Required env vars (export before running or prefix on CLI):
#     PROJECT_ROOT : absolute path to your Moto checkout
#     OUTPUT_ROOT  : where raw + lmdb data should live
#
#  Example:
#     export PROJECT_ROOT=/workspace/Moto
#     export OUTPUT_ROOT=/workspace/calvin_data
#     bash download_and_preprocess_calvin_split.sh task_D_D
# ------------------------------------------------------------
set -euo pipefail

# ------------ CONFIG ---------------------------------------------------------
SPLIT="${1:-task_D_D}"                    # default split name; override with arg
PROJECT_ROOT="${PROJECT_ROOT:?Need to export PROJECT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Need to export OUTPUT_ROOT}"

export TOKENIZERS_PARALLELISM=false # add to silence the warning
ACCELERATE_ARGS="--data.num_workers 2"   # extra CLI flags

RAW_DIR="${OUTPUT_ROOT}/${SPLIT}"         # where raw files land
LMDB_DIR="${OUTPUT_ROOT}/lmdb_datasets/${SPLIT}"
DATA_URL="http://calvin.cs.uni-freiburg.de/dataset/${SPLIT}.zip"
SCENE_URL="http://calvin.cs.uni-freiburg.de/scene_info_fix/${SPLIT}_scene_info.zip"

mkdir -p "${OUTPUT_ROOT}"
cd "${OUTPUT_ROOT}"

echo "[CALVIN] ###############################################################"
echo "[CALVIN] Processing split: ${SPLIT}"
echo "[CALVIN] Output root       : ${OUTPUT_ROOT}"
echo "[CALVIN] Project root      : ${PROJECT_ROOT}"
echo "-----------------------------------------------------------------------"

# ------------ 1) DEMONSTRATIONS --------------------------------------------
if [ -d "${RAW_DIR}" ]; then
    echo "[CALVIN] Raw dataset already present → ${RAW_DIR}"
else
    echo "[CALVIN] Downloading demonstrations zip ..."
    wget -c "${DATA_URL}" -O "${SPLIT}.zip"
    echo "[CALVIN] Unzipping ..."
    unzip -q "${SPLIT}.zip"
    rm -f  "${SPLIT}.zip"
    echo "[CALVIN] Saved to ${RAW_DIR}"
fi

# ------------ 2) SCENE INFO -------------------------------------------------
cd "${RAW_DIR}"
if [ -f transforms.json ] || [ -f camera_static.yaml ]; then
    echo "[CALVIN] Scene-info already merged into ${RAW_DIR}"
else
    echo "[CALVIN] Downloading scene-info zip ..."
    wget -c "${SCENE_URL}" -O "${SPLIT}_scene_info.zip"
    unzip -q -o "${SPLIT}_scene_info.zip"
    rm -f   "${SPLIT}_scene_info.zip"
    echo "[CALVIN] Scene-info merged."
fi
cd "${OUTPUT_ROOT}"

# ------------ 3) LMDB CONVERSION -------------------------------------------
if [ -d "${LMDB_DIR}" ]; then
    echo "[CALVIN] LMDB already exists → ${LMDB_DIR}"
else
    echo "[CALVIN] Converting to LMDB ..."
    mkdir -p "$(dirname "${LMDB_DIR}")"
    python3 "${PROJECT_ROOT}/data_preprocessing/calvin_to_lmdb.py" \
        --input_dir  "${RAW_DIR}" \
        --output_dir "${LMDB_DIR}"
    echo "[CALVIN] LMDB created at ${LMDB_DIR}"
fi

echo "[CALVIN] ####################  DONE  ###################################"
