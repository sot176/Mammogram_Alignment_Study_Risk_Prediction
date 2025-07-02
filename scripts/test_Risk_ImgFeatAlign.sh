#!/bin/bash -l


# Define placeholder variables
CSV_FILE_PATH="PATH_TO_CSV_FILE"
DATA_ROOT_PATH="PATH_TO_DATA_ROOT"
TRAIN_OUTPUT_PATH="PATH_TO_TRAINING_OUTPUT_DIRECTORY"
TEST_OUTPUT_PATH="PATH_TO_TEST_OUTPUT_DIRECTORY"
TRAINING_ID="YOUR_TRAINING_ID"
DATASET="EMBED"  # or "CSAW"
BATCH_SIZE="YOUR_BATCH_SIZE"
SEED=2023
CONTAINER="PATH_TO_YOUR_SINGULARITY_CONTAINER"

mkdir -p "$TEST_OUTPUT_PATH"

srun singularity exec \
    $CONTAINER \
    torchrun --standalone \
             --nnodes=1 \
             --nproc-per-node=${SLURM_GPUS_PER_NODE} \
    main_test_risk_prediction.py \
    --csv_file "$CSV_FILE_PATH" \
    --data_root "$DATA_ROOT_PATH" \
    --path_out_dir "$TRAIN_OUTPUT_PATH" \
    --path_test_folder "$TEST_OUTPUT_PATH" \
    --id_training "$TRAINING_ID" \
    --batch_size "$BATCH_SIZE" \
    --use_img_alignment "True" \
    --use_img_feat_alignment "True" \
    --no_feat_Alignment "False" \
    --use_implicit_alignment "False" \
    --early_stop "False" \
    --dataset "$DATASET" \
    --seed "$SEED"

