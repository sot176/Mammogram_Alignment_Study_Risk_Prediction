import logging
import os
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.utils import resample
from matplotlib.colors import TwoSlopeNorm

from utils.c_index import concordance_index_ipcw



def get_risk_loss_BCE( pred, y_true, y_mask):
    """
    Binary cross-entropy loss adapted for cumulative risk prediction with masking.
    Args:
        pred: Logits for cumulative risk, tensor of shape [B, T]
        y_true: Binary ground truth labels, tensor of shape [B, T]
                (1 if event happened by year t)
        y_mask: Mask tensor of shape [B, T], where 1 indicates valid data for year t
                and 0 indicates censored or invalid data

    Returns:
        masked_loss: Scalar tensor representing the masked binary cross-entropy loss.
    """
    y_mask = y_mask.to(pred.device)
    y_true = y_true.to(pred.device)
    masked_loss = F.binary_cross_entropy_with_logits(
        pred, y_true.float(), weight=y_mask.float(), size_average=False
    ) / torch.sum(y_mask.float())

    return masked_loss

def normalize_feature_map(feature_map):
    """
    Normalize feature maps using channel-wise mean and standard deviation.

    Args:
        feature_map: Tensor of shape [B, C, H, W] representing feature maps.

    Returns:
        normalized_map: Tensor of same shape as input, normalized per channel.
    """
    mean = feature_map.mean(dim=(2, 3), keepdim=True)
    std = feature_map.std(dim=(2, 3), keepdim=True)
    normalized_map = (feature_map - mean) / (std + 1e-8)  # epsilon for numerical stability
    return normalized_map

def checkpoint(model, filename):
    """
    Save model state dictionary to a file.

    Args:
        model: PyTorch model whose state_dict will be saved.
        filename: String path to save the model state dictionary.
    """
    torch.save(model.state_dict(), filename)


def create_logger(log_path):
    """
    Create a logger that writes INFO-level logs to a specified file.

    Args:
        log_path: Path to the log file.

    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    f_handler = logging.FileHandler(log_path)
    f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(f_handler)
    return logger


def bootstrap_auc_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    n_bootstrap=1000,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for AUC by density categories.

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        density_categories: Array of categorical density labels (N,), values in ["A", "B", "C", "D"]
        n_bootstrap: Number of bootstrap samples (default=1000)
        alpha: Significance level for confidence intervals (default=0.05)

    Returns:
        auc_summary_by_density: Dict mapping density categories to dicts with year keys and
                               values as (mean_auc, (lower_CI, upper_CI))
    """
    auc_results_by_density = {d: {f"Year {i+1}": [] for i in range(5)} for d in ["A", "B", "C", "D"]}

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]
        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        cancer_indices = np.where(event_observed_density == 1)[0]
        non_cancer_indices = np.where(event_observed_density == 0)[0]

        if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
            print(f"Skipping density '{density}' due to missing cancer or non-cancer cases.")
            continue

        for _ in range(n_bootstrap):
            cancer_sample = resample(cancer_indices, replace=True, n_samples=len(cancer_indices))
            non_cancer_sample = resample(non_cancer_indices, replace=True, n_samples=len(non_cancer_indices))
            indices = np.concatenate([cancer_sample, non_cancer_sample])

            event_times_sample = event_times_density[indices]
            predictions_sample = predictions_density[indices]
            event_observed_sample = event_observed_density[indices]

            yearly_aucs_sample = compute_auc_x_year_auc(
                predictions_sample, event_times_sample, event_observed_sample
            )

            for year, auc in yearly_aucs_sample.items():
                auc_results_by_density[density][f"Year {year + 1}"].append(auc)

    auc_summary_by_density = {}
    for density, auc_results in auc_results_by_density.items():
        auc_summary_by_density[density] = {}
        for year, auc_values in auc_results.items():
            if auc_values:
                lower = np.percentile(auc_values, 100 * alpha / 2)
                upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
                auc_summary_by_density[density][year] = (np.mean(auc_values), (lower, upper))
            else:
                auc_summary_by_density[density][year] = (None, (None, None))

    return auc_summary_by_density


def bootstrap_c_index(
    event_times,
    predictions,
    event_observed,
    censoring_dist,
    n_bootstrap=1000,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for concordance index.

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        censoring_dist: Censoring distribution for IPCW calculation
        n_bootstrap: Number of bootstrap samples (default=1000)
        alpha: Significance level for confidence intervals (default=0.05)

    Returns:
        mean_c_index: Mean concordance index over bootstrap samples
        ci: Tuple of lower and upper confidence interval bounds
    """
    c_index_scores = []

    cancer_indices = np.where(event_observed == 1)[0]
    non_cancer_indices = np.where(event_observed == 0)[0]

    for _ in range(n_bootstrap):
        cancer_sample = resample(cancer_indices, replace=True, n_samples=len(cancer_indices))
        non_cancer_sample = resample(non_cancer_indices, replace=True, n_samples=len(non_cancer_indices))
        indices = np.concatenate([cancer_sample, non_cancer_sample])

        event_times_sample = event_times[indices]
        predictions_sample = predictions[indices]
        event_observed_sample = event_observed[indices]

        c_index = concordance_index_ipcw(
            event_times_sample,
            predictions_sample,
            event_observed_sample,
            censoring_dist,
        )
        c_index_scores.append(c_index)

    lower = np.percentile(c_index_scores, 100 * alpha / 2)
    upper = np.percentile(c_index_scores, 100 * (1 - alpha / 2))

    return np.mean(c_index_scores), (lower, upper)


def bootstrap_c_index_by_density(
    event_times,
    predictions,
    event_observed,
    density_categories,
    censoring_dist,
    n_bootstrap=1000,
    alpha=0.05,
    save_json_path=None,
):
    """
    Compute bootstrap confidence intervals for concordance index by density categories,
    optionally saving the bootstrap samples to a JSON file.

    Args:
        event_times: Array of event/censoring times (N,)
        predictions: Array of predicted risk scores (N,)
        event_observed: Binary array indicating event occurrence (N,)
        density_categories: Array of categorical density labels (N,), values in ["A", "B", "C", "D"]
        censoring_dist: Censoring distribution for IPCW calculation
        n_bootstrap: Number of bootstrap samples (default=1000)
        alpha: Significance level for confidence intervals (default=0.05)
        save_json_path: Optional path to save bootstrap results JSON file (default=None)

    Returns:
        c_index_summary_by_density: Dict mapping density categories to
                                   (mean_c_index, (lower_CI, upper_CI)) tuples.
    """

    c_index_results_by_density = {density: [] for density in ["A", "B", "C", "D"]}

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]
        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        cancer_indices = np.where(event_observed_density == 1)[0]
        non_cancer_indices = np.where(event_observed_density == 0)[0]

        if len(cancer_indices) == 0 or len(non_cancer_indices) == 0:
            print(f"Skipping density '{density}' due to missing cancer or non-cancer cases.")
            continue

        for _ in range(n_bootstrap):
            cancer_sample = resample(cancer_indices, replace=True, n_samples=len(cancer_indices))
            non_cancer_sample = resample(non_cancer_indices, replace=True, n_samples=len(non_cancer_indices))
            indices = np.concatenate([cancer_sample, non_cancer_sample])

            event_times_sample = event_times_density[indices]
            predictions_sample = predictions_density[indices]
            event_observed_sample = event_observed_density[indices]

            c_index = concordance_index_ipcw(
                event_times_sample,
                predictions_sample,
                event_observed_sample,
                censoring_dist,
            )
            c_index_results_by_density[density].append(c_index)

    if save_json_path is not None:
        filename = "mbox_plots_c_index_density_results.json"
        file_path = os.path.join(save_json_path, filename)
        c_index_serializable = {
            density: list(map(float, values))
            for density, values in c_index_results_by_density.items()
        }
        with open(file_path, "w") as f:
            json.dump(c_index_serializable, f)
        print(f"[INFO] Saved bootstrap C-index samples to {file_path}")

    c_index_summary_by_density = {}
    for density, c_index_values in c_index_results_by_density.items():
        if c_index_values:
            lower = np.percentile(c_index_values, 100 * alpha / 2)
            upper = np.percentile(c_index_values, 100 * (1 - alpha / 2))
            c_index_summary_by_density[density] = (np.mean(c_index_values), (lower, upper))
        else:
            c_index_summary_by_density[density] = (None, (None, None))

    return c_index_summary_by_density


def bootstrap_confidence_interval(data, num_samples=1000, confidence_level=0.95):
    """
    Calculate the confidence interval using bootstrapping.

    Args:
        data: List or numpy array of metric values.
        num_samples: Number of bootstrap samples to draw (default: 1000).
        confidence_level: Confidence level for the interval (default: 0.95).

    Returns:
        Tuple (lower_bound, upper_bound) representing the confidence interval.
    """
    data = np.array(data)
    bootstrapped_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1 - confidence_level
    lower_bound = np.percentile(bootstrapped_means, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound


def bootstrap_auc(event_times, predictions, event_observed, n_bootstrap=1000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for AUC at 5 yearly follow-ups.

    Args:
        event_times: Array of event/censoring times (N,).
        predictions: Array of predicted risk probabilities or scores (N, T).
        event_observed: Binary array indicating event occurrence (N,).
        n_bootstrap: Number of bootstrap samples (default: 1000).
        alpha: Significance level for confidence intervals (default: 0.05).

    Returns:
        auc_summary: Dict mapping "Year 1" to "Year 5" to tuples of (mean_auc, (lower_CI, upper_CI)).
    """
    auc_results = {f"Year {i + 1}": [] for i in range(5)}

    cancer_indices = np.where(event_observed == 1)[0]
    non_cancer_indices = np.where(event_observed == 0)[0]

    for _ in range(n_bootstrap):
        cancer_sample = resample(cancer_indices, replace=True, n_samples=len(cancer_indices))
        non_cancer_sample = resample(non_cancer_indices, replace=True, n_samples=len(non_cancer_indices))
        indices = np.concatenate([cancer_sample, non_cancer_sample])

        event_times_sample = event_times[indices]
        predictions_sample = predictions[indices]
        event_observed_sample = event_observed[indices]

        yearly_aucs_sample = compute_auc_x_year_auc(predictions_sample, event_times_sample, event_observed_sample)
        for year, auc in yearly_aucs_sample.items():
            auc_results[f"Year {year + 1}"].append(auc)

    auc_summary = {}
    for year, auc_values in auc_results.items():
        lower = np.percentile(auc_values, 100 * alpha / 2)
        upper = np.percentile(auc_values, 100 * (1 - alpha / 2))
        auc_summary[year] = (np.mean(auc_values), (lower, upper))

    return auc_summary

def print_results(results):
    """
    Nicely print nested dictionaries or key-value pairs.

    Args:
        results: Dict or nested dict to print.
    """
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


def compute_auc_x_year_auc(probs, censor_times, golds):
    """
    Compute AUC for each year from 1 to 5 given predicted probabilities, censoring times, and event labels.

    Args:
        probs: List or array of predicted probabilities with shape (N, 5).
        censor_times: Array of censoring/event times (N,).
        golds: Array of binary event indicators (N,).

    Returns:
        aucs_per_year: Dict mapping follow-up year (0 to 4) to AUC values.
    """
    def include_exam_and_determine_label(prob_arr, censor_time, gold, followup):
        valid_pos = gold == 1 and censor_time <= followup  # event occurred before or at followup
        valid_neg = censor_time >= followup
        included = valid_pos or valid_neg
        label = valid_pos
        return included, label

    aucs_per_year = {}

    for followup in range(5):
        probs_for_eval, golds_for_eval = [], []
        for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
            include, label = include_exam_and_determine_label(prob_arr, censor_time, gold, followup)
            if include:
                probs_for_eval.append(prob_arr[followup])
                golds_for_eval.append(label)

        try:
            auc = metrics.roc_auc_score(golds_for_eval, probs_for_eval, average="samples")
        except Exception as e:
            warnings.warn(f"Failed to calculate AUC because {e}")
            auc = "NA"
        aucs_per_year[followup] = auc

    return aucs_per_year


def compute_auc_by_density_category(predictions, event_times, event_observed, density_categories):
    """
    Compute AUC for each density category (A, B, C, D) and each follow-up year.

    Args:
        predictions: List or array of predicted probabilities (N, 5).
        event_times: List or array of event/censor times (N,).
        event_observed: List or array of event indicators (N,).
        density_categories: List or array of density categories (N,), values in {"A","B","C","D"}.

    Returns:
        aucs_by_density: Dict with keys 'A','B','C','D', each mapping to a dict of yearly AUCs.
    """
    aucs_by_density = {"A": {}, "B": {}, "C": {}, "D": {}}

    for density in ["A", "B", "C", "D"]:
        idx = [i for i, cat in enumerate(density_categories) if cat == density]
        probs = [predictions[i] for i in idx]
        event_times_filtered = [event_times[i] for i in idx]
        event_observed_filtered = [event_observed[i] for i in idx]

        aucs_by_density[density] = compute_auc_x_year_auc(probs, event_times_filtered, event_observed_filtered)

    return aucs_by_density


def compute_c_index_by_density(event_times, predictions, event_observed, density_categories, censoring_dist):
    """
    Compute the concordance index (C-index) for each density category without bootstrapping.

    Args:
        event_times: Array of event/censoring times (N,).
        predictions: Array of predicted risk scores (N,).
        event_observed: Array of binary event indicators (N,).
        density_categories: Array of density categories (N,), values in {"A","B","C","D"}.
        censoring_dist: Censoring distribution used for IPCW calculation.

    Returns:
        c_indexes_by_density: Dict mapping density categories to their corresponding C-index values.
    """
    c_indexes_by_density = {"A": None, "B": None, "C": None, "D": None}

    for density in ["A", "B", "C", "D"]:
        density_indices = np.where(density_categories == density)[0]
        event_times_density = event_times[density_indices]
        predictions_density = predictions[density_indices]
        event_observed_density = event_observed[density_indices]

        try:
            c_index = concordance_index_ipcw(
                event_times_density,
                predictions_density,
                event_observed_density,
                censoring_dist,
            )
        except Exception as e:
            print(f"Error calculating C-index for density {density}: {e}")
            c_index = None

        c_indexes_by_density[density] = c_index

    return c_indexes_by_density

def save_model_results_to_file(probs, censor_times, golds, density_categories, censoring_dist, out_dir):
    """
    Computes and saves the AUC values, C-index by density, predictions,
    censoring times, and event labels to a JSON file.

    Args:
        probs: Array of predicted probabilities (N, T).
        censor_times: Array of event or censoring times (N,).
        golds: Array of event indicators (1 if event occurred, else 0) (N,).
        density_categories: Array of breast density categories (N,), values in {"A", "B", "C", "D"}.
        censoring_dist: Censoring distribution used for IPCW calculation.
        out_dir: Directory path to save the output JSON file.

    Saves:
        model_results.json containing:
            - "C_index_by_density": Concordance index per density category.
            - "auc_per_year": AUC values for years 1–5.
            - "predictions": Predicted probabilities (as list).
            - "censor_times": Event/censor times (as list).
            - "golds": Ground truth labels (as list).
    """
    # Compute AUC per year for the current model
    aucs_per_year = compute_auc_x_year_auc(probs, censor_times, golds)

    # Compute C-index by density
    c_indexes_by_density = compute_c_index_by_density(
        censor_times,
        probs,
        golds,
        density_categories,
        censoring_dist,
    )

    # Prepare results dictionary
    results_dict = {
        "C_index_by_density": c_indexes_by_density,
        "auc_per_year": aucs_per_year,
        "predictions": probs.tolist(),
        "censor_times": censor_times.tolist(),
        "golds": golds.tolist()
    }

    # Define output file path
    filename = "model_results.json"
    file_path = os.path.join(out_dir, filename)

    # Save the results to a JSON file
    with open(file_path, 'w') as file:
        json.dump(results_dict, file, indent=4)
        print(f"Results for all models saved to {file_path}")
def NJD_std(displacement):
    """
    Compute the standard deviation of Jacobian determinants from a displacement field.

    Args:
        displacement: Displacement field of shape (B, H, W, 2).

    Returns:
        float: Standard deviation of the Jacobian determinant values.
    """
    D_y = displacement[:, 1:, :-1, :] - displacement[:, :-1, :-1, :]
    D_x = displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1, :]
    D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
    D2 = D_x[..., 1] * D_y[..., 0]
    Ja_value = D1 - D2
    std_value = np.std(Ja_value)
    return std_value


def NJD_percentage(displacement):
    """
    Calculate the percentage of negative Jacobian determinants.

    Args:
        displacement: Displacement field of shape (B, H, W, 2).

    Returns:
        float: Percentage of negative Jacobian determinants.
    """
    D_y = displacement[:, 1:, :-1, :] - displacement[:, :-1, :-1, :]
    D_x = displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1, :]
    D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
    D2 = D_x[..., 1] * D_y[..., 0]
    Ja_value = D1 - D2
    percentage = 100.0 * (np.sum(Ja_value < 0) / np.sum(Ja_value))
    return percentage


class NJD:
    """
    Class for computing the Jacobian determinant of a 2D displacement field.

    Args:
        Lambda: Regularization term (default: 1e-5).
    """

    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda

    def get_Ja(self, displacement):
        """
        Compute the Jacobian determinant.

        Args:
            displacement: Displacement field of shape (B, H, W, 2).

        Returns:
            numpy array: Jacobian determinant of shape (B, H-1, W-1).
        """
        D_y = displacement[:, 1:, :-1, :] - displacement[:, :-1, :-1, :]
        D_x = displacement[:, :-1, 1:, :] - displacement[:, :-1, :-1, :]
        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
        D2 = D_x[..., 1] * D_y[..., 0]
        Ja_value = D1 - D2
        return Ja_value

    def get_Ja_plot(self, displacement):
        """
        Compute the Jacobian determinant using central difference for visualization.

        Args:
            displacement: Displacement field of shape (B, H, W, 2).

        Returns:
            numpy array: Jacobian determinant of shape (B, H-2, W-2).
        """
        D_y = (displacement[:, 2:, 1:-1, :] - displacement[:, :-2, 1:-1, :]) / 2
        D_x = (displacement[:, 1:-1, 2:, :] - displacement[:, 1:-1, :-2, :]) / 2
        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1)
        D2 = D_x[..., 1] * D_y[..., 0]
        Ja_value = D1 - D2
        return Ja_value


def plot_deformation_field(fix_id, mov_id, njd, out_dir, displacement_field_np, J_det):
    """
    Visualize and save the deformation field and Jacobian determinant heatmap.

    Args:
        fix_id: Filename or ID of the fixed image.
        mov_id: Filename or ID of the moving image.
        njd: NJD percentage value to annotate the plot.
        out_dir: Directory to save the output plot.
        displacement_field_np: Displacement field of shape (H, W, 2).
        J_det: Jacobian determinant values of shape (H, W).

    Saves:
        A PNG image visualizing the deformation field and Jacobian determinant.
    """
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    f = plt.figure(figsize=(18, 12))
    fig_name = "Deformation_field_Fixed_{}_moving_{}.png".format(fix_id[:-4], mov_id[-8:-4])

    ax6 = f.add_subplot(121)
    ax7 = f.add_subplot(122)

    step = 2
    H, W = displacement_field_np.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step]

    deformation_field_downsampled = displacement_field_np[::step, ::step]
    u = deformation_field_downsampled[..., 0]
    v = deformation_field_downsampled[..., 1]

    ax6.quiver(x, y, u, v, color="red")
    ax6.axis("off")
    ax6.set_aspect("equal")
    ax6.set_title("Deformation Field")

    vmin = -1
    vmax = 2
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)

    img = ax7.imshow(np.squeeze(J_det), cmap="RdBu", interpolation="none", norm=norm)
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=26)
    cbar.ax.tick_params(width=2)

    ax7.axis("off")
    ax7.set_title("Jacobian Determinant of Displacement Field")

    plt.suptitle("NJD (%): " + str(njd), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, fig_name))
    plt.close()