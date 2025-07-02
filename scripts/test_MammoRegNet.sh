#!/bin/bash -l

# Define placeholder variables
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
TRAINING_OUTPUT_PATH="PATH_TO_TRAINING_OUTPUT_DIRECTORY"
TEST_OUTPUT_PATH="PATH_TO_TEST_OUTPUT_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
BATCH_SIZE="YOUR_BATCH_SIZE"
NUM_WORKERS="YOUR_NUM_WORKERS"
DATASET="EMBED"  # or "CSAW"
CONTAINER="PATH_TO_YOUR_SINGULARITY_CONTAINER"

# Create test output directory
mkdir -p "$TEST_OUTPUT_PATH"

# Run the testing script using Singularity
srun singularity exec \
    "$CONTAINER" \
    torchrun --standalone \
             --nnodes=1 \
             --nproc-per-node="${SLURM_GPUS_PER_NODE}" \
    main_test_mammoregnet.py \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$TRAINING_OUTPUT_PATH" \
    --path_test_folder "$TEST_OUTPUT_PATH" \
    --id_training "$TRAINING_ID" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --dataset "$DATASET"