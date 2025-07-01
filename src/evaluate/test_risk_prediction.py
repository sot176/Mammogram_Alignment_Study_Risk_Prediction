import torch
import torch.nn.functional as F

import numpy as np

from utils.utils import *
from utils.c_index import get_censoring_dist
from models import model_combined_alignment_risk, mammoregnet


def test_jointly_feat_alignment_risk(
    test_loader,
    device,
    path_model,
    out_dir,
    path_logger,
    no_feat_Alignment,
    use_implicit_alignment
):
    """
       Evaluate a risk prediction model (with or without feature alignment) on the test set.

       Args:
           test_loader: DataLoader for the test dataset.
           device: CUDA or CPU device.
           path_model: Path to the saved model.
           out_dir: Directory to store deformation field visualizations.
           path_logger: Path for the log file.
           no_feat_Alignment: "True" to disable feature alignment.
           use_implicit_alignment: "True" to enable implicit alignment mode.

       Returns:
           Dictionary of evaluation metrics including C-index, AUC, NJD.
       """
    logger = create_logger(path_logger)
    print("[INFO] Loading trained risk model...")

    # Select model class based on configuration
    if no_feat_Alignment == "True":
        model_cls = model_combined_alignment_risk.RiskModel_implicit_alignment_Mirai if use_implicit_alignment == "True" \
            else model_combined_alignment_risk.RiskModel_no_alignment_Mirai
    else:
        model_cls = model_combined_alignment_risk.CombinedAlignmentRiskModel_Mirai

    # Initialize model
    model_risk = model_cls()
    checkpoint = torch.load(path_model, map_location=device)
    model_risk.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
    model_risk.to(device).eval()

    print("[INFO] Evaluating on test dataset...")

    # Tracking variables
    predictions, event_times, event_observed, density_categories = [], [], [], []
    njd_values, njd_std_values = [], []
    test_running_njd_value, test_running_njd_std, counter = 0.0, 0.0, 0

    with torch.inference_mode():
        for batch in test_loader:
            torch.cuda.empty_cache()

            img_curr = batch["current_image"].to(device, dtype=torch.float32)
            img_prev = batch["previous_image"].to(device, dtype=torch.float32)
            time_gap = batch["time_gap"].to(device)
            event_time = batch["event_times"].to(device, dtype=torch.float32)
            event_obs = batch["event_observed"].to(device, dtype=torch.float32)
            density = batch["density"]

            # Forward pass
            output = model_risk(img_curr, img_prev, time_gap)
            risk_pred = output["risk_prediction"]["pred_fused"]
            predictions.append(torch.sigmoid(risk_pred).detach().cpu().numpy())
            event_times.append(event_time.cpu().numpy())
            event_observed.append(event_obs.cpu().numpy())
            density_categories.append(density)

            # Deformation field evaluation (only when alignment is enabled)
            if no_feat_Alignment != "True":
                flow = output["deformation_field"]
                for i in range(flow.size(0)):
                    counter += 1
                    df = flow[i].unsqueeze(0).detach().cpu().permute(0, 2, 3, 1).numpy()

                    njd_std = NJD_std(df)
                    jac_det = NJD().get_Ja_plot(df).numpy()

                    njd_values.append(njd.item())
                    njd_std_values.append(njd_std)
                    test_running_njd_value += njd
                    test_running_njd_std += njd_std

                    plot_deformation_field(
                        batch["current_image_id"][i],
                        batch["previous_image_id"][i],
                        njd,
                        out_dir,
                        df.squeeze(),
                        jac_det,
                    )

    # Concatenate data
    predictions = np.concatenate(predictions, axis=0)
    event_times = np.concatenate(event_times, axis=0)
    event_observed = np.concatenate(event_observed, axis=0)
    density_categories = np.concatenate(density_categories, axis=0)

    # Censoring distribution for IPCW
    censoring_dist = get_censoring_dist(event_times, event_observed)

    # Save predictions and labels
    save_model_results_to_file(predictions, event_times, event_observed, density_categories, censoring_dist, out_dir)

    print("[INFO] Calculating metrics...")

    # C-index
    mean_c_index, c_index_ci = bootstrap_c_index(event_times, predictions, event_observed, censoring_dist)

    # Yearly AUC
    auc_summary = bootstrap_auc(event_times, predictions, event_observed)
    auc_by_density = bootstrap_auc_by_density(event_times, predictions, event_observed, density_categories)
    c_index_by_density = bootstrap_c_index_by_density(
        event_times, predictions, event_observed, density_categories, censoring_dist, save_json_path=out_dir
    )

    auc_formatted = {
        f"{year}": {"Mean": mean_auc, "95% CI": ci}
        for year, (mean_auc, ci) in auc_summary.items()
    }

    # NJD metrics
    results = {
        "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
        "Yearly AUCs": auc_formatted,
        "AUC by density categories": auc_by_density,
        "C index by density categories": c_index_by_density,
    }

    if no_feat_Alignment != "True":
        njd_mean = test_running_njd_value / counter
        njd_ci = bootstrap_confidence_interval(np.array(njd_values))

        njd_std_mean = test_running_njd_std / counter
        njd_std_ci = bootstrap_confidence_interval(np.array(njd_std_values))

        results.update({
            "NJD": {"Mean": njd_mean, "95% CI": njd_ci},
            "Std dev Jacobian": {"Mean": njd_std_mean, "95% CI": njd_std_ci},
        })

    # Logging
    logger.info(f"[RESULTS] Evaluation Summary:\n{results}")
    print({"Results": results})

    return results


def test_img_alignment_risk_pred_combined_train(
    test_loader,
    device,
    path_model,
    out_dir,
    path_logger,
    use_img_feat_alignment,
    dataset
):
    """
    Evaluate the trained model on the test dataset and compute C-index and AUC for years 1 to 5.
    Parameters:
    - test_loader: DataLoader for the test dataset.
    - device: Device to use for computation (e.g., 'cuda' or 'cpu').
    - path_model: Path to the trained model file.
    - out_dir: Directory to save outputs such as plots.
    - path_logger: Path to the logger file.
    - use_img_alignment_way1_downsampled: Flag to determine the model type.
    Returns:
    - results: A dictionary containing C-index and AUC scores for each year.
    """

    print("[INFO] Loading the trained model...")
    logger = create_logger(path_logger)

    # Load registration model
    model_reg = mammoregnet.MammoRegNet()
    if dataset == "CSAW":
        # Example path for CSAW dataset registration model
        path_saved_reg_model = "/path/to/csaw/registration_models/best_model_registration.pth"
    else:
        # Example path for other dataset registration model
        path_saved_reg_model = "/path/to/other/registration_models/best_model_registration.pth"
    print("Path reg model", path_saved_reg_model)

    checkpoint_reg = torch.load(path_saved_reg_model, map_location=device, weights_only=True)
    checkpoint_reg = {k.replace("module.", ""): v.to("cpu") for k, v in checkpoint_reg.items()}

    model_reg.load_state_dict(checkpoint_reg)
    model_reg = model_reg.to(device).eval()

    # Load risk model
    model_cls = (
        model_combined_alignment_risk.CombinedImgAlignmentRiskModel_downsample_img_deformation_field_Mirai
        if use_img_feat_alignment == "True"
        else model_combined_alignment_risk.CombinedImgAlignmentRiskModel_Mirai
    )
    model_risk = model_cls()

    checkpoint = torch.load(path_model, map_location=device)
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model_risk.load_state_dict(checkpoint)
    model_risk = model_risk.to(device).eval()

    print("[INFO] Evaluating on test dataset...")

    # Evaluation Loop
    predictions, event_times, event_observed, density_categories = [], [], [], []
    njd_values, njd_std_values = [], []
    test_running_njd_value, test_running_njd_std, counter = 0.0, 0.0, 0.0

    with torch.inference_mode():
        for batch in test_loader:
            torch.cuda.empty_cache()

            # Load data
            curr_img = batch["current_image"].to(device, dtype=torch.float32)
            prev_img = batch["previous_image"].to(device, dtype=torch.float32)
            time_gap = batch["time_gap"].to(device)
            if dataset in {"EMBED", "CSAW"}:
                density_batch = batch["density"]

            # Registration model forward
            warped_img, def_field = model_reg(curr_img, prev_img)

            # Risk model forward
            outputs = model_risk(curr_img, prev_img, warped_img, def_field, time_gap)
            risk_pred_fused = outputs["risk_prediction"]["pred_fused"]
            def_field_out = outputs["deformation_field"]

            # Store outputs
            predictions.append(torch.sigmoid(risk_pred_fused).cpu().numpy())
            event_observed.append(batch["event_observed"].cpu().numpy())
            event_times.append(batch["event_times"].cpu().numpy())
            density_categories.append(density_batch)

            # Process deformation field (NJD + plots)
            for i in range(def_field_out.shape[0]):
                counter += 1

                df = def_field_out[i].unsqueeze(0).cpu()
                df_down = F.interpolate(df, size=(64, 52), mode="bilinear", align_corners=True)
                df_down[:, 0, :, :] *= 52 / 1664  # x
                df_down[:, 1, :, :] *= 64 / 2048  # y
                df_reshaped = df_down.permute(0, 2, 3, 1)

                img_id = batch["current_image_id"][i]
                if "_R_" in img_id:
                    df_crop = df_reshaped[:, 4:-4, 4:, :]
                elif "_L_" in img_id:
                    df_crop = df_reshaped[:, 4:-4, :-4, :]
                else:
                    df_crop = df_reshaped  # fallback

                njd = NJD_percentage(df_crop.numpy())
                njd_std = NJD_std(df_crop.numpy())
                jac_det = NJD().get_Ja_plot(df_crop.numpy())

                njd_values.append(njd)
                njd_std_values.append(njd_std)
                test_running_njd_value += njd
                test_running_njd_std += njd_std

                plot_deformation_field(
                    batch["current_image_id"][i],
                    batch["previous_image_id"][i],
                    njd,
                    out_dir,
                    df_crop.squeeze().numpy(),
                    jac_det,
                )

            del curr_img, prev_img, warped_img, def_field, time_gap, outputs

    # Metrics
    njd_mean = test_running_njd_value / counter
    njd_std_mean = test_running_njd_std / counter
    njd_ci = bootstrap_confidence_interval(np.array(njd_values))
    njd_std_ci = bootstrap_confidence_interval(np.array(njd_std_values))

    predictions = np.concatenate(predictions, axis=0)
    event_times = np.concatenate(event_times, axis=0)
    event_observed = np.concatenate(event_observed, axis=0)
    density_categories = np.concatenate(density_categories, axis=0)

    censoring_dist = get_censoring_dist(event_times, event_observed)

    save_model_results_to_file(predictions, event_times, event_observed, density_categories, censoring_dist, out_dir)

    print("[INFO] Calculating AUC for each year and C-index ...")
    mean_c_index, c_index_ci = bootstrap_c_index(event_times, predictions, event_observed, censoring_dist)
    auc_summary = bootstrap_auc(event_times, predictions, event_observed)
    auc_by_density = bootstrap_auc_by_density(event_times, predictions, event_observed, density_categories)
    c_index_by_density = bootstrap_c_index_by_density(
        event_times, predictions, event_observed, density_categories, censoring_dist, save_json_path=out_dir
    )

    auc_formatted = {
        f"{year}": {"Mean": mean_auc, "95% CI": ci}
        for year, (mean_auc, ci) in auc_summary.items()
    }

    results = {
        "C-index": {"Mean": mean_c_index, "95% CI": c_index_ci},
        "Yearly AUCs": auc_formatted,
        "AUC by density categories": auc_by_density,
        "C index by density categories": c_index_by_density,
        "NJD": {"Mean": njd_mean, "95% CI": njd_ci},
        "Std dev Jacobian": {"Mean": njd_std_mean, "95% CI": njd_std_ci},
    }

    logger.info(f"Results: {results}")
    print_results(results)

    return results