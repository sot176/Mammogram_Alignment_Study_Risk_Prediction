#!/bin/bash -l

# Define placeholder variables for paths and arguments
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
OUTPUT_DIR_PATH="PATH_TO_OUTPUT_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="EMBED"  # or "CSAW"
BATCH_SIZE="YOUR_BATCH_SIZE"
NUM_WORKERS="YOUR_NUM_WORKERS"
LEARNING_RATE="YOUR_LEARNING_RATE"
CONTAINER="PATH_TO_YOUR_SINGULARITY_CONTAINER"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"

# Run the training script using Singularity
srun singularity exec \
    $CONTAINER \
    torchrun --standalone \
             --nnodes=1 \
             --nproc-per-node=${SLURM_GPUS_PER_NODE} \
    main_train_mammoregnet.py \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$OUTPUT_DIR_PATH" \
    --id_training "$TRAINING_ID" \
    --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --learning_rate "$LEARNING_RATE"
