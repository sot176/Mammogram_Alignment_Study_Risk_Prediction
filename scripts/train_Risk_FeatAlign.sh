#!/bin/bash -l

# Define placeholder variables
CSV_FILE_PATH="PATH_TO_CSV_FILE"
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
BASE_OUTPUT_PATH="PATH_TO_OUTPUT_BASE_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="EMBED"  # or "CSAW"
BATCH_SIZE="YOUR_BATCH_SIZE"
NUM_WORKERS="YOUR_NUM_WORKERS"
LEARNING_RATE="YOUR_LEARNING_RATE"
WEIGHT_DECAY="YOUR_WEIGHT_DECAY"
LR_DECAY="YOUR_LR_DECAY"
NUM_EPOCHS="YOUR_NUM_EPOCHS"
SEED=2023
CONTAINER="PATH_TO_YOUR_SINGULARITY_CONTAINER"


# Construct output directory path with SLURM job info
OUTPUT_DIR_PATH="${BASE_OUTPUT_PATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR_PATH"

# Run the training script using Singularity
srun singularity exec \
    "$CONTAINER" \
    torchrun --standalone \
             --nnodes=1 \
             --nproc-per-node="${SLURM_GPUS_PER_NODE}" \
    main_train_risk_prediction.py \
    --csv_file "$CSV_FILE_PATH" \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$OUTPUT_DIR_PATH" \
    --id_training "$TRAINING_ID" \
    --batch_size "$BATCH_SIZE" \
    --augmentations "True" \
    --use_scheduler "True" \
    --use_img_alignment "False" \
    --use_img_feat_alignment "False" \
    --use_reg_loss "False" \
    --no_feat_Alignment "False" \
    --use_implicit_alignment "False" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --lr_decay "$LR_DECAY" \
    --num_epochs "$NUM_EPOCHS" \
    --dataset "$DATASET" \
    --seed "$SEED"





