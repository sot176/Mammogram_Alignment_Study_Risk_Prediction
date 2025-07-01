import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.c_index import concordance_index_ipcw, get_censoring_dist
from models.model_combined_alignment_risk import ( RiskModel_no_alignment_Mirai, RiskModel_implicit_alignment_Mirai, \
                                                  CombinedAlignmentRiskModel_Mirai, CombinedImgAlignmentRiskModel_Mirai, \
                                                   CombinedImgAlignmentRiskModel_downsample_img_deformation_field_Mirai)
from models.mammoregnet import MammoRegNet
from utils.utils import get_risk_loss_BCE, compute_auc_x_year_auc, create_logger, Regu_loss



# function to generate the output from only rank zero
def rank_zero_tqdm(iterable, **kwargs):
    if dist.get_rank() == 0:
        return tqdm(iterable, **kwargs)
    else:
        return iterable  # Other ranks skip tqdm


# function to get the model size
def get_model_size(model):
    # Handle DDP wrapping
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    if dist.get_rank() == 0:  # Only log from main process
        print(f"Model size: {total_size_mb:.2f} MB (Parameters: {param_size / 1e6:.1f}M elements)")
    return total_size_mb


def train_val_jointly(
    args,
    train_loader,
    valid_loader,
    learning_rate,
    weight_decay,
    num_epochs,
    path_loggger,
    path_model,
    id,
    use_scheduler,
    out_dir,
    patience_lr_scheduler,
    patience,
    use_reg_loss,
    lambda_regu,
    lr_decay,
    no_feat_Alignment,
    local_rank,
    use_implicit_alignment,
):
    # Device and DDP setup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = (
        RiskModel_implicit_alignment_Mirai if use_implicit_alignment == "True"
        else RiskModel_no_alignment_Mirai if no_feat_Alignment == "True"
        else CombinedAlignmentRiskModel_Mirai
    )
    model_risk = model_cls().to(local_rank)
    model_risk = DDP(model_risk, device_ids=[local_rank])

    # Logger
    logger = create_logger(path_loggger)
    if dist.get_rank() == 0:
        logger.info(f"[INFO] Number of Training Epochs: {num_epochs}")

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        [p for p in model_risk.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = None
    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=lr_decay, patience=patience_lr_scheduler, verbose=True
        )
        if dist.get_rank() == 0:
            logger.info(f"Scheduler Initialized: {type(scheduler).__name__}")

        # WandB
    if dist.get_rank() == 0:
        wandb.init(
            project="EMBED_Risk_Prediction",
            config={
                "optimizer": optimizer.__class__.__name__,
                "architecture": "TemporalRiskPrediction",
                "dataset": "EMBED",
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model": model_risk.__class__.__name__,
            },
        )
        wandb.define_metric("epoch", hidden=True)
        for metric in [
            "Training Loss", "Training Alignment Loss", "Training Risk Loss", "Training C-index",
            "Validation Risk Loss", "Validation Alignment L2 Loss", "Validation C-index",
            "Year 1 AUC", "Year 2 AUC", "Year 3 AUC", "Year 4 AUC", "Year 5 AUC"
        ]:
            wandb.define_metric(metric, step_metric="epoch")

        # Loss Functions
    alignment_loss_fn = nn.MSELoss()
    Loss_regu = Regu_loss

    best_c_index = 0
    patience_counter = 0

    for epoch in rank_zero_tqdm(range(num_epochs), desc="Epochs"):
        if dist.get_rank() == 0:
            logger.info(f"##### Epoch: {epoch} #####")

        model_risk.train()
        running = {
            "loss": 0.0,
            "alignment_loss": 0.0,
            "risk_loss": 0.0,
        }

        counter = 0
        all_preds, all_times, all_events = [], [], []

        for idx, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            counter += 1

            current_image = batch["current_image"].to(local_rank, dtype=torch.float32)
            prior_image = batch["previous_image"].to(local_rank, dtype=torch.float32)
            time_gap = batch["time_gap"].to(local_rank)
            target = batch["target"]
            target_prior = batch["target_prior"]
            y_mask = batch["y_mask"]
            y_mask_prior = batch["y_mask_prior"]
            event_times = batch["event_times"].to(local_rank, dtype=torch.float32)
            event_observed = batch["event_observed"].to(local_rank, dtype=torch.float32)

            # Forward pass
            outputs = model_risk(current_image, prior_image, time_gap)
            del current_image, prior_image, time_gap

            pred = outputs["risk_prediction"]
            fused = pred["pred_fused"]
            cur = pred["pred_cur"]
            pri = pred["pred_pri"]

            # Risk Loss
            risk_loss_fused = get_risk_loss_BCE(fused, target, y_mask)
            risk_loss_cur = get_risk_loss_BCE(cur, target, y_mask)
            risk_loss_pri = get_risk_loss_BCE(pri, target_prior, y_mask_prior)

            risk_loss = risk_loss_fused + risk_loss_cur + risk_loss_pri
            running["risk_loss"] += risk_loss.item()

            # Alignment Loss
            if no_feat_Alignment != "True":
                aligned = outputs["aligned_prior_feature"]
                current = outputs["current_feature"]
                alignment_loss = alignment_loss_fn(aligned, current)
                running["alignment_loss"] += alignment_loss.item()

            # Regularization Loss
            if no_feat_Alignment == "True":
                total_loss = risk_loss
            else:
                regu_loss = Loss_regu(outputs["deformation_field"]) if use_reg_loss == "True" else 0
                total_loss = risk_loss + (alignment_loss / 10) + (
                    lambda_regu * regu_loss if use_reg_loss == "True" else 0)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running["loss"] += total_loss.item()

            all_preds.append(torch.sigmoid(fused).detach().cpu().numpy())
            all_times.append(event_times.detach().cpu().numpy())
            all_events.append(event_observed.detach().cpu().numpy())

        # Evaluation
        preds = np.concatenate(all_preds, axis=0)
        times = np.concatenate(all_times, axis=0)
        events = np.concatenate(all_events, axis=0)
        censoring = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring)

        # Averages
        avg_loss = running["loss"] / counter
        avg_align = running["alignment_loss"] / counter
        avg_risk = running["risk_loss"] / counter

        if dist.get_rank() == 0:
            wandb.log({
                "epoch": epoch,
                "Training Loss": avg_loss,
                "Training Alignment Loss": avg_align,
                "Training Risk Loss": avg_risk,
                "Training C-index": c_index,
            })
            logger.info(
                f"[Epoch {epoch}] Total Loss: {avg_loss:.4f} | Alignment Loss: {avg_align:.4f} | Risk Loss: {avg_risk:.4f}"
            )

        ##################      Validation      ###################
        with torch.no_grad():
            model_risk.eval()

            running = {
                "loss": 0.0,
                "alignment_loss": 0.0,
                "alignment_loss_before": 0.0,
                "risk_loss": 0.0,
            }

            predictions, event_observed, event_times = [], [], []
            counter = 0

            for idx, batch_val in enumerate(valid_loader):
                torch.cuda.empty_cache()
                counter += 1

                current_image_val = batch_val["current_image"].to(local_rank, dtype=torch.float32)
                prior_image_val = batch_val["previous_image"].to(local_rank, dtype=torch.float32)
                time_gap_val = batch_val["time_gap"].to(local_rank, dtype=torch.float32)
                event_times_batch = batch_val["event_times"].to(local_rank, dtype=torch.float32)
                event_observed_batch = batch_val["event_observed"].to(local_rank, dtype=torch.float32)
                y_mask_val = batch_val["y_mask"]
                y_mask_val_prior = batch_val["y_mask_prior"]
                target_prior_val = batch_val["target_prior"]
                target_val = batch_val["target"]

                # Forward pass
                outputs_val = model_risk(current_image_val, prior_image_val, time_gap_val)
                del current_image_val, prior_image_val, time_gap_val

                # Alignment Loss
                if no_feat_Alignment != "True":
                    aligned_prior_val = outputs_val["aligned_prior_feature"]
                    current_features_val = outputs_val["current_feature"]
                    prior_feature = outputs_val["prior_feature_before_alignment"]

                    alignment_loss_val = alignment_loss_fn(aligned_prior_val, current_features_val)
                    alignment_loss_val_before = alignment_loss_fn(prior_feature, current_features_val)

                    running["alignment_loss"] += alignment_loss_val.item()
                    running["alignment_loss_before"] += alignment_loss_val_before.item()

                # Risk Loss
                risk_pred_val = outputs_val["risk_prediction"]
                risk_loss_fused = get_risk_loss_BCE(risk_pred_val["pred_fused"], target_val, y_mask_val)
                risk_loss_cur = get_risk_loss_BCE(risk_pred_val["pred_cur"], target_val, y_mask_val)
                risk_loss_pri = get_risk_loss_BCE(risk_pred_val["pred_pri"], target_prior_val, y_mask_val_prior)
                risk_loss = risk_loss_fused + risk_loss_cur + risk_loss_pri
                running["risk_loss"] += risk_loss.item()

                # Total Loss
                total_loss = risk_loss if no_feat_Alignment == "True" else risk_loss + (alignment_loss_val / 10)
                running["loss"] += total_loss.item()

                # Collect predictions and event data
                predictions.append(torch.sigmoid(risk_pred_val["pred_fused"]).detach().cpu().numpy())
                event_times.append(event_times_batch.detach().cpu().numpy())
                event_observed.append(event_observed_batch.detach().cpu().numpy())

                # Metrics
            predictions = np.concatenate(predictions, axis=0)
            event_times = np.concatenate(event_times, axis=0)
            event_observed = np.concatenate(event_observed, axis=0)

            censoring_dist = get_censoring_dist(event_times, event_observed)
            c_index = concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist)
            auc_results = compute_auc_x_year_auc(predictions, event_times, event_observed)

            torch.cuda.empty_cache()

            # Logging (rank 0 only)
            if dist.get_rank() == 0:
                avg_loss = running["loss"] / counter
                avg_alignment_loss = running["alignment_loss"] / counter
                avg_risk_loss = running["risk_loss"] / counter

                # Log AUCs per year
                for year, auc in auc_results.items():
                    logger.info(f"Year {year + 1}: AUC = {auc:.6f}")
                    wandb.log({f"Year {year + 1} AUC": auc, "epoch": epoch})

                # Log main metrics
                wandb.log({
                    "Validation Total Loss": avg_loss,
                    "Validation Alignment L2 Loss": avg_alignment_loss,
                    "Validation Risk Loss": avg_risk_loss,
                    "Validation C-index": c_index,
                    "epoch": epoch,
                })

                logger.info(
                    f"[Validation] Total Loss: {avg_loss:.4f} | Alignment Loss: {avg_alignment_loss:.4f} | Risk Loss: {avg_risk_loss:.4f}")
                logger.info(f"[Validation] C-index: {c_index:.4f}")

                # Scheduler
                if use_scheduler == "True":
                    scheduler.step(c_index)

                # Checkpoint saving
                if epoch % 10 == 0:
                    torch.save(model_risk.state_dict(), os.path.join(out_dir, f'checkpoint{epoch:04}.pth'))

                # Early stopping
                if c_index > best_c_index:
                    best_c_index = c_index
                    patience_counter = 0
                    torch.save(model_risk.state_dict(),
                               os.path.join(out_dir, f"best_model_risk_prediction_id-{id}.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stop_flag = torch.tensor(1, device=local_rank)
                        dist.broadcast(early_stop_flag, src=0)

                        if early_stop_flag.item() == 1:
                            logger.info("Early stopping triggered.")
                            torch.save(model_risk.state_dict(),
                                       os.path.join(out_dir, f"early_stopping_risk_prediction_id-{id}.pth"))
                            dist.barrier()
                            break

    if dist.get_rank() == 0:
        # Save the trained model
        print("[INFO] Saving model ...")
        torch.save(model_risk.state_dict(), path_model)

        # Log the model to W&B
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(path_model)
        wandb.log_artifact(artifact)

        # Finish W&B logging
        wandb.finish()

    # Cleanup distributed process group
    dist.barrier()  # Ensure all processes have finished before continuing
    dist.destroy_process_group()


def train_val_jointly_img_alignment(
    args,
    train_loader,
    valid_loader,
    learning_rate,
    weight_decay,
    num_epochs,
    path_loggger,
    path_model,
    id,
    use_scheduler,
    out_dir,
    patience_lr_scheduler,
    patience,
    use_img_feat_alignment,
    lr_decay,
    dataset, local_rank
):

    # initialize logger
    logger = create_logger(path_loggger)
    logger.info("Number Training Epochs: {}".format(num_epochs))
    if dataset == "CSAW":
        # Example path for CSAW dataset registration model
        path_saved_reg_model = "/path/to/csaw/registration_models/best_model_registration.pth"
    else:
        # Example path for other dataset registration model
        path_saved_reg_model = "/path/to/other/registration_models/best_model_registration.pth"
    print("Path reg model", path_saved_reg_model)

    # Load registration checkpoint, map weights to CPU first
    checkpoint = torch.load(path_saved_reg_model, map_location="cpu", weights_only=True)
    # Remove 'module.' prefix if present (for DataParallel or DDP)
    new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    # Initialize registration model and load weights
    model_reg = MammoRegNet()
    model_reg.load_state_dict(new_checkpoint)
    model_reg = model_reg.to(local_rank)
    model_reg.eval()

    # Initialize risk prediction model with registration model passed in constructor
    model_risk = (
        CombinedImgAlignmentRiskModel_downsample_img_deformation_field_Mirai(num_years=5, registration_model=model_reg)
        if use_img_feat_alignment == "True"
        else CombinedImgAlignmentRiskModel_Mirai(num_years=5, registration_model=model_reg)
    )
    model_risk = model_risk.to(local_rank  )

    # Wrap model with DDP
    model_risk = DDP(model_risk, device_ids=[local_rank  ])

    get_model_size(model_risk)

    # Prepare optimizer
    params_to_optimize = [p for p in model_risk.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

    # Setup learning rate scheduler if used
    if use_scheduler == "True":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_decay,
            patience=patience_lr_scheduler,
            verbose=True,
        )
        if dist.get_rank() == 0:
            logger.info(f"Scheduler configured: {type(scheduler).__name__}")

    # Initialize WandB (only on rank 0)
    if dist.get_rank() == 0:
        wandb.init(
            project="EMBED_Risk_Prediction",
            config={
                "Optimizer": optimizer,
                "architecture": "TemporalRiskPrediction",
                "dataset": dataset,
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "Weight_decay": weight_decay,
                "Model": model_risk,
            },
        )

        wandb.define_metric("epoch", hidden=True)
        for metric in [
            "Training Loss", "Training Alignment Loss", "Training Risk Loss", "Training C-index",
            "Validation Risk Loss", "Validation Alignment L2 Loss", "Validation C-index",
            "Year 1 AUC", "Year 2 AUC", "Year 3 AUC", "Year 4 AUC", "Year 5 AUC"
        ]:
            wandb.define_metric(metric, step_metric="epoch")

    # prepare losses
    best_c_index = 0
    patience_counter = 0
    alignment_loss_fn = nn.MSELoss()  # L2 loss for alignment

    for epoch in rank_zero_tqdm(range(num_epochs), desc="Epochs"):
        if dist.get_rank() == 0:
            logger.info(f"##### Epoch: {epoch} #####")

        model_risk.train()

        running = {
            "risk_loss": 0.0,
        }
        counter = 0
        all_preds, all_times, all_events = [], [], []

        for idx, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            counter += 1

            # Load batch inputs and move to device
            current_image = batch["current_image"].to(local_rank, dtype=torch.float32)
            prior_image = batch["previous_image"].to(local_rank, dtype=torch.float32)
            time_gap = batch["time_gap"].to(local_rank)
            target = batch["target"]
            target_prior = batch["target_prior"]
            y_mask = batch["y_mask"]
            y_mask_prior = batch["y_mask_prior"]
            event_times_batch = batch["event_times"].to(local_rank)
            event_observed_batch = batch["event_observed"].to(local_rank)

            with torch.no_grad():
                registration_outputs = model_reg(current_image, prior_image)

            warped_pri_img = registration_outputs[0]
            deformation_field = registration_outputs[1]

            # Forward pass through risk model
            outputs = model_risk(current_image, prior_image, warped_pri_img.to(local_rank),
                                 deformation_field.to(local_rank), time_gap)

            del prior_image, current_image, registration_outputs, warped_pri_img, deformation_field

            # Extract risk predictions
            risk_pred = outputs["risk_prediction"]
            fused = risk_pred["pred_fused"]
            cur = risk_pred["pred_cur"]
            pri = risk_pred["pred_pri"]

            # Compute risk losses
            risk_loss_fused = get_risk_loss_BCE(fused, target, y_mask)
            risk_loss_cur = get_risk_loss_BCE(cur, target, y_mask)
            risk_loss_pri = get_risk_loss_BCE(pri, target_prior, y_mask_prior)

            risk_loss = risk_loss_fused + risk_loss_cur + risk_loss_pri
            running["risk_loss"] += risk_loss.item()

            optimizer.zero_grad()
            risk_loss.backward()
            optimizer.step()
            del risk_loss

            all_preds.append(torch.sigmoid(fused).detach().cpu().numpy())
            all_times.append(event_times_batch.detach().cpu().numpy())
            all_events.append(event_observed_batch.detach().cpu().numpy())

            # Concatenate all batches for evaluation
        preds = np.concatenate(all_preds, axis=0)
        times = np.concatenate(all_times, axis=0)
        events = np.concatenate(all_events, axis=0)
        censoring_dist = get_censoring_dist(times, events)
        c_index = concordance_index_ipcw(times, preds, events, censoring_dist)

        # Compute averages
        avg_risk_loss = running["risk_loss"] / counter

        # Logging only on rank 0
        if dist.get_rank() == 0:
            wandb.log({
                "epoch": epoch,
                "Training Risk Loss": avg_risk_loss,
                "Training C-index": c_index,
            })

            logger.info(f"Risk Loss: {avg_risk_loss:.4f}")
            logger.info(f"Training C-index: {c_index:.4f}")

        ##################      Validation      ###################

        with torch.no_grad():
            running = {
                "loss": 0.0,
                "alignment_loss": 0.0,
                "risk_loss": 0.0,
            }

            predictions, event_times, event_observed = [], [], []
            counter = 0

            for idx, batch_val in enumerate(valid_loader):
                torch.cuda.empty_cache()
                counter += 1

                current_image_val = batch_val["current_image"].to(local_rank, dtype=torch.float32)
                prior_image_val = batch_val["previous_image"].to(local_rank, dtype=torch.float32)
                time_gap_val = batch_val["time_gap"].to(local_rank).to(torch.int8)
                event_times_batch = batch_val["event_times"].to(local_rank).to(torch.int8)
                event_observed_batch = batch_val["event_observed"].to(local_rank).to(torch.int8)
                y_mask_val = batch_val["y_mask"]
                y_mask_val_prior = batch_val["y_mask_prior"]
                target_prior_val = batch_val["target_prior"]
                target_val = batch_val["target"]

                # Registration model forward pass
                registration_outputs_val = model_reg(current_image_val, prior_image_val)
                warped_pri_img_val = registration_outputs_val[0]
                deformation_field_val = registration_outputs_val[1]

                # Main model forward pass
                outputs_val = model_risk(
                    current_image_val, prior_image_val, warped_pri_img_val, deformation_field_val, time_gap_val
                )
                del current_image_val, prior_image_val, registration_outputs_val, deformation_field_val, warped_pri_img_val

                # Alignment Loss
                aligned_prior_val = outputs_val["aligned_prior_feature"]
                current_features_val = outputs_val["current_feature"]
                alignment_loss_val = alignment_loss_fn(aligned_prior_val, current_features_val)
                running["alignment_loss"] += alignment_loss_val.item()

                # Risk prediction loss
                risk_prediction_val = outputs_val["risk_prediction"]
                risk_pred_fused = risk_prediction_val["pred_fused"]
                risk_pred_cur = risk_prediction_val["pred_cur"]
                risk_pred_pri = risk_prediction_val["pred_pri"]
                del risk_prediction_val

                risk_loss_fused = get_risk_loss_BCE(risk_pred_fused, target_val, y_mask_val)
                risk_loss_cur = get_risk_loss_BCE(risk_pred_cur, target_val, y_mask_val)
                risk_loss_pri = get_risk_loss_BCE(risk_pred_pri, target_prior_val, y_mask_val_prior)
                risk_loss_val = risk_loss_fused + risk_loss_cur + risk_loss_pri

                del risk_loss_fused, risk_loss_cur, risk_loss_pri
                running["risk_loss"] += risk_loss_val.item()

                # Total loss
                total_loss = risk_loss_val + (alignment_loss_val / 10)
                running["loss"] += total_loss.item()

                # Collect predictions and events
                predictions.append(torch.sigmoid(risk_pred_fused).detach().cpu().numpy())
                event_times.append(event_times_batch.cpu().numpy())
                event_observed.append(event_observed_batch.cpu().numpy())

                # Aggregate predictions and metrics
            predictions = np.concatenate(predictions, axis=0)
            event_times = np.concatenate(event_times, axis=0)
            event_observed = np.concatenate(event_observed, axis=0)
            censoring_dist = get_censoring_dist(event_times, event_observed)

            c_index = concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist)
            auc_results = compute_auc_x_year_auc(predictions, event_times, event_observed)

            alignment_loss_epoch = running["alignment_loss"] / counter
            risk_loss_epoch = running["risk_loss"] / counter
            avg_loss = running["loss"] / counter

            torch.cuda.empty_cache()

            if dist.get_rank() == 0:
                # Log AUC per year
                for year, auc in auc_results.items():
                    logger.info(f"Year {year + 1}: AUC = {auc:.6f}")
                    wandb.log({f"Year {year + 1} AUC": auc, "epoch": epoch})

                # Log main metrics
                wandb.log({
                    "Validation Total Loss": avg_loss,
                    "Validation Alignment L2 Loss": alignment_loss_epoch,
                    "Validation Risk Loss": risk_loss_epoch,
                    "Validation C-index": c_index,
                    "epoch": epoch,
                })

                logger.info(
                    f"[Validation] Total Loss: {avg_loss:.4f} | Alignment Loss: {alignment_loss_epoch:.4f} | Risk Loss: {risk_loss_epoch:.4f}"
                )
                logger.info(f"[Validation] C-index: {c_index:.4f}")

                if use_scheduler == "True":
                    scheduler.step(c_index)

                # Save checkpoints every 10 epochs
                if epoch % 10 == 0:
                    torch.save(model_risk.state_dict(), os.path.join(out_dir, f'checkpoint{epoch:04}.pth'))

                # Early stopping
                if c_index > best_c_index:
                    best_c_index = c_index
                    patience_counter = 0
                    torch.save(model_risk.state_dict(),
                               os.path.join(out_dir, f"best_model_risk_prediction_id-{id}.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stop_flag = torch.tensor(1, device=local_rank)
                        dist.broadcast(early_stop_flag, src=0)

                        if early_stop_flag.item() == 1:
                            logger.info("Early stopping triggered.")
                            torch.save(model_risk.state_dict(),
                                       os.path.join(out_dir, f"early_stopping_risk_prediction_id-{id}.pth"))
                            dist.barrier()
                            break

    if dist.get_rank() == 0:
        # Save the trained model
        print("[INFO] Saving model ...")
        torch.save(model_risk.state_dict(), path_model)

        # Log the model to W&B
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(path_model)
        wandb.log_artifact(artifact)

        # Finish W&B logging
        wandb.finish()

        # Cleanup distributed process group
    dist.barrier()  # Ensure all processes have fini
    dist.destroy_process_group()
