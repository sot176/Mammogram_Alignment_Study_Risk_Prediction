import argparse
import os
import random
import torch
from kornia.constants import Resample
from torch.utils.data import DataLoader
from dataloaders.dataset import BreastCancerRiskDataset
from dataloaders.dataset_csaw import BreastCancerRiskDatasetCSAWCC
from train.train_risk_prediction import train_val_jointly, train_val_jointly_img_alignment
import logging
import time
import kornia.augmentation as K_A
from kornia.constants import Resample
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


# to get rid of warning on the slurm output.
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)



def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")


# function to log the details
def setup_logging(rank, path_logger):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank == 0:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler (writes to log file)
        file_handler = logging.FileHandler(path_logger, mode="w")  # Overwrite
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        # Console handler (prints to stdout)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training config for breast cancer risk prediction")

    # Paths and dataset
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with dataset info")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset images")
    parser.add_argument("--path_out_dir", type=str, required=True, help="Output directory for saving models and logs")
    parser.add_argument("--id_training", type=int, required=True, help="Unique training run ID")
    parser.add_argument("--dataset", type=str, default="EMBED", help="Dataset to use (EMBED or CSAW)")

    # Training settings
    parser.add_argument("--augmentations", type=str, required=True, help="Enable data augmentation if 'True'")
    parser.add_argument("--use_scheduler", type=str, required=True, help="Use learning rate scheduler if 'True'")
    parser.add_argument("--use_img_alignment", type=str, default="False", help="Enable image-level alignment if 'True'")
    parser.add_argument("--use_img_feat_alignment", type=str, default="False",
                        help="Enable image-feature alignment if 'True'")
    parser.add_argument("--no_feat_Alignment", type=str, default="False", help="Disable feature alignment if 'True'")
    parser.add_argument("--use_reg_loss", type=str, default="False", help="Use regularization loss if 'True'")
    parser.add_argument("--lambda_regu", type=float, default=0.2, help="Weight for regularization loss")

    parser.add_argument("--patience_lr_scheduler", default=5, type=int, help="Patience epochs for LR scheduler")
    parser.add_argument("--patience", default=15, type=int, help="Patience epochs for early stopping")
    parser.add_argument("--accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr_decay", default=0.5, type=float, help="Learning rate decay factor")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")

    # DataLoader params
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training and validation")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--schuffle", default=True, type=bool, help="Shuffle training data")
    parser.add_argument("--pin_memory", default=True, type=bool, help="Use pin_memory in DataLoader")

    # Reproducibility
    parser.add_argument("--seed", default=2023, type=int, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    ddp_setup()
    args = parse_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.use_class_weights == "True":
        print("Using class weights for the BCE loss")
    # Define datasets and dataloader
    if args.augmentations == "True":  ### For newest and oldest mammograms
        transforms_img_train =torch.nn.Sequential(
            K_A.RandomAffine(translate=(0.0, 0.1), scale=(1.0, 1.05), degrees=0, shear=0, p=0.5),
            K_A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0, p=0.5),
            K_A.RandomGamma(gamma=(0.8, 1.4), gain=(0.95, 1.05), p=0.5),
            K_A.RandomCrop(size=(1843, 1498), p=0.2),
            K_A.Resize((2048, 1664)),
        )
        print("Augmentations", transforms_img_train)
        transforms_img_val = None
    else:
        transforms_img_train = None
        transforms_img_val = None

    print("Train augmentations", transforms_img_train)
    if args.dataset == "CSAW":
        train_dataset = BreastCancerRiskDatasetCSAWCC(
            args.csv_file, args.data_root, "train", transforms=transforms_img_train
        )
        validation_dataset = BreastCancerRiskDatasetCSAWCC(
            args.csv_file, args.data_root, "val", transforms=transforms_img_val
        )
    else:
        train_dataset = BreastCancerRiskDataset(
            args.csv_file, args.data_root, "train", transforms=transforms_img_train
        )
        validation_dataset = BreastCancerRiskDataset(
            args.csv_file, args.data_root, "val", transforms=transforms_img_val
        )

    # Dataloader setup with DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(validation_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        sampler=valid_sampler
    )

    # Define paths for model checkpoint and logging
    model_path = f"model_risk_prediction_training_id_{args.id_training}_last_epoch.pth"
    log_path = f"train_risk_prediction_training_id_{args.id_training}.log"
    path_out_model = os.path.join(args.path_out_dir, model_path)
    path_logger = os.path.join(args.path_out_dir, log_path)

    print(f"Model will be saved to: {path_out_model}")
    print(f"Log will be saved to: {path_logger}")

    # Ensure the directory exists
    os.makedirs(args.path_out_dir, exist_ok=True)

    # call the logging
    rank = dist.get_rank()
    logger = setup_logging(rank, path_logger)

    start_time = time.time()

    if args.use_img_alignment == "True":
        if dist.get_rank() == 0:
            logger.info("Training started with  ImgAlign or ImgFeatAlign from train_val_jointly_img_alignment function ...")
        train_val_jointly_img_alignment(args,
            train_loader,
            validation_loader,
            args.learning_rate,
            args.weight_decay,
            args.num_epochs,
            path_logger,
            path_out_model,
            args.id_training,
            args.use_scheduler,
            args.path_out_dir,
            args.patience_lr_scheduler,
            args.patience,
            args.use_img_feat_alignment,
            args.lr_decay,
            args.dataset, rank
        )

    else:
        if dist.get_rank() == 0:
            logger.info("Training started with  FeatAlign, FeatAlignReg or NoAlign from train_val_jointly function...")
        train_val_jointly(args,
            train_loader,
            validation_loader,
            args.learning_rate,
            args.weight_decay,
            args.num_epochs,
            path_logger,
            path_out_model,
            args.id_training,
            args.use_scheduler,
            args.path_out_dir,
            args.patience_lr_scheduler,
            args.patience,
            args.use_reg_loss,
            args.lambda_regu,
            args.lr_decay,
            args.no_feat_Alignment,rank, args.use_implicit_alignment
        )

    end_time = time.time()
    if dist.get_rank() == 0:
        logger.info(f"Training completed in {(end_time - start_time)/60:.2f}minutes")
        logger.info(f"Saving model to: {path_out_model}")


if __name__ == '__main__':
    main()
