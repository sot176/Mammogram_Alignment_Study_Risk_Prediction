import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F

from models.mammoregnet import (AffineTransformer_block, MammoRegNet,
                                    SpatialTransformer_block)
from train.losses_mammoregnet import NCC, Regu_loss, NJD
from utils.utils import *




def normalize(arr):
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) *255.0 / rng



def plot_images(
    img_fix,
    img_mov,
    warped_img,
    affine_img,
    fix_id,
    mov_id,
    ncc_before,
    ncc_final,
    ncc_affine,
    njd,
    out_dir,
):
    plt.figure(figsize=(18, 12))
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)  # This creates the directory if it doesn't already exist

    fig_name = "Fixed image{}_moving image{}.png".format(fix_id[:-4], mov_id[:-4])

    plt.subplot(231)
    plt.imshow(img_fix.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Fixed image_{}".format(fix_id[:-4]))

    plt.subplot(232)
    plt.imshow(img_mov.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Moving image_{}".format(mov_id[:-4]))

    plt.subplot(233)
    plt.imshow(warped_img.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Warped moving image")

    plt.subplot(234)
    plt.imshow(img_fix.squeeze().cpu().numpy(), cmap="gray")
    plt.imshow(img_mov.squeeze().cpu().numpy(), cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Fixed image overlayed with moving image")

    plt.subplot(235)
    plt.imshow(img_fix.squeeze().cpu().numpy(), cmap="gray")
    plt.imshow(affine_img.squeeze().cpu().numpy(), cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Fixed image overlayed with affine transformed moving image")

    plt.subplot(236)
    plt.imshow(img_fix.squeeze().cpu().numpy(), cmap="gray")
    plt.imshow(warped_img.squeeze().cpu().numpy(), cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Fixed image overlayed with final transformed moving image")

    plt.suptitle(
        "NCC before: "
        + "  "
        + str(round(ncc_before, 6))
        + " , "
        + " NCC after: "
        + str(round(ncc_final, 6))
        + " , "
        + " NCC only affine: "
        + str(round(ncc_affine, 6))
        + " , "
        " NJD (%): " + str(njd),
        fontsize=10,
    )
    plt.tight_layout()

    plt.savefig(os.path.join(images_dir, fig_name))
    plt.close()


def plot_deformation_field_jac_det(
    fix_id, mov_id, njd, out_dir, displacement_field_np, J_det
):
    """
    Plot deformation field vectors and the Jacobian determinant side-by-side.

    Args:
        fix_id (str): Identifier for the fixed image.
        mov_id (str): Identifier for the moving image.
        njd (float): Normalized Jacobian determinant metric (NJD percentage).
        out_dir (str): Directory to save the output figure.
        displacement_field_np (np.ndarray): Displacement field array of shape (H, W, 2).
        J_det (np.ndarray): Jacobian determinant array of shape (H, W).
    """
    plt.figure(figsize=(18, 12))

    # Construct filename based on IDs (strip extensions)
    fig_name = f"Deformation_field_Fixed_{fix_id[:-4]}_moving_{mov_id[-8:-4]}.png"

    # Create subplots side-by-side
    ax_deform = plt.subplot(1, 2, 1)
    ax_jacobian = plt.subplot(1, 2, 2)

    # Downsampling step for quiver plot clarity
    step = 2
    H, W = displacement_field_np.shape[:2]

    # Create meshgrid for plotting vectors
    y, x = np.mgrid[0:H:step, 0:W:step]

    # Downsample displacement field for visualization
    deformation_downsampled = displacement_field_np[::step, ::step]

    # Extract vector components (u: x-direction, v: y-direction)
    u = deformation_downsampled[..., 0]
    v = deformation_downsampled[..., 1]

    # Plot deformation vectors as quiver plot
    ax_deform.quiver(x, y, u, v, color="red", angles="xy", scale_units="xy", scale=1)
    ax_deform.axis("off")
    ax_deform.set_aspect("equal")
    ax_deform.set_title("Deformation Field", fontsize=14)

    # Plot Jacobian determinant heatmap
    vmin, vmax = -2, 2  # Symmetric range around zero for color scale
    img = ax_jacobian.imshow(
        np.squeeze(J_det), cmap="RdBu", interpolation="nearest", vmin=vmin, vmax=vmax
    )

    # Create colorbar axis aligned to the right of the heatmap
    divider = make_axes_locatable(ax_jacobian)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add colorbar with formatting
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=12, width=1.5)

    ax_jacobian.axis("off")
    ax_jacobian.set_title("Jacobian Determinant of Displacement Field", fontsize=14)

    # Add a main title with NJD percentage
    plt.suptitle(f"NJD (%): {njd:.2f}", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save and close the figure
    plt.savefig(os.path.join(out_dir, fig_name))
    plt.close()


def test(test_loader, device, path_saved_model, path_logger, out_dir):
    # initialize logger
    logger = create_logger(path_logger)

    # Load trained model
    model = MammoRegNet()
    checkpoint = torch.load(path_saved_model, map_location=device)  # Use "cuda" if needed
    new_checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_checkpoint)
    model.to(device).eval()

    # Loss functions
    ncc_loss = NCC().loss
    jacobian_analyzer = NJD()

    ncc_before_total, ncc_final_total, ncc_affine_total, njd_total, njd_std_total = 0.0, 0.0, 0.0, 0.0, 0.0
    ncc_final_list, ncc_affine_list, njd_list, njd_std_list = [], [], [], []
    test_running_njd_value, test_running_njd_std, counter = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            for idx in range(batch["img_fix"].shape[0]):
                counter += 1

                img_fix = batch["img_fix"][idx].to(device)
                img_mov = batch["img_mov"][idx].to(device)
                fix_id = batch["img_fix_id"][idx]
                mov_id = batch["img_mov_id"][idx]

                # Forward pass
                start_time = time.time()
                pred = model(img_fix.unsqueeze(0), img_mov.unsqueeze(0))
                time_elapsed = time.time() - start_time
                time_total += time_elapsed

                warped_img = pred[0][0]
                flow_field = pred[1][0]
                affine_img = pred[2][0]

                # Compute NCC metrics
                ncc_before = ncc_loss(img_fix, img_mov).item()
                ncc_final = ncc_loss(img_fix, warped_img).item()
                ncc_affine = ncc_loss(img_fix, affine_img).item()

                ncc_before_total += ncc_before
                ncc_final_total += ncc_final
                ncc_affine_total += ncc_affine
                ncc_final_list.append(ncc_final)
                ncc_affine_list.append(ncc_affine)

                # Downsample and scale flow for NJD
                flow_down = F.interpolate(flow_field.unsqueeze(0).cpu(), size=(32, 16), mode="bilinear",
                                          align_corners=True)
                flow_down[:, 0, :, :] *= 16 / 512  # width scale
                flow_down[:, 1, :, :] *= 32 / 1024  # height scale
                flow_np = flow_field.unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy()
                flow_down_np = flow_down.permute(0, 2, 3, 1).squeeze(0).numpy()

                # Compute NJD and its std
                njd_value = NJD_percentage(flow_np).item()
                njd_std = NJD_std(flow_np).item()

                njd_total += njd_value
                njd_list.append(njd_value)
                njd_std_total += njd_std
                njd_std_list.append(njd_std)

                # Generate visualizations
                warped_np = warped_img.squeeze(0).cpu().numpy()
                warped_pil = Image.fromarray(warped_np.astype(np.uint8))
                warped_pil.save(os.path.join(out_dir, f"{mov_id[:-4]}_warped.png"))

                jacobian = jacobian_analyzer.get_Ja(flow_down_np)
                plot_deformation_field_jac_det(fix_id, mov_id, njd_value, out_dir, flow_down_np, jacobian)
                plot_images(img_fix, img_mov, warped_img, affine_img, fix_id, mov_id, ncc_before, ncc_final, ncc_affine,
                            njd_value, out_dir)

        # Compute averages
        avg_ncc_before = ncc_before_total / counter
        avg_ncc_final = ncc_final_total / counter
        avg_ncc_affine = ncc_affine_total / counter
        avg_njd = njd_total / counter
        avg_njd_std = njd_std_total / counter

        results = {
            "NJD": {
                "Mean": avg_njd,
                "95% CI": bootstrap_confidence_interval(np.array(njd_list)),
            },
            "Std dev Jacobian": {
                "Mean": avg_njd_std,
                "95% CI": bootstrap_confidence_interval(np.array(njd_std_list)),
            },
            "NCC Before": avg_ncc_before,
            "NCC Only Affine": {
                "Mean": avg_ncc_affine,
                "95% CI": bootstrap_confidence_interval(np.array(ncc_affine_list)),
            },
            "Final NCC": {
                "Mean": avg_ncc_final,
                "95% CI": bootstrap_confidence_interval(np.array(ncc_final_list)),
            },
        }

        # Logging
        logger.info(f"Number of image pairs: {counter}")
        logger.info(f"Average NCC before registration: {avg_ncc_before}")
        logger.info(f"Average NCC (final): {avg_ncc_final}")
        logger.info(f"Average NCC (affine only): {avg_ncc_affine}")
        logger.info(f"Average NJD: {avg_njd}")
        logger.info(f"Average NJD Std: {avg_njd_std}")
        logger.info(f"Results with confidence intervals: {results}")

        # Console output
        print(f"\n--- Test Summary ({counter} image pairs) ---")
        print(f"Average NCC Before       : {avg_ncc_before:.4f}")
        print(f"Average NCC Final        : {avg_ncc_final:.4f}")
        print(f"Average NCC Affine Only  : {avg_ncc_affine:.4f}")
        print(f"Average NJD              : {avg_njd:.2f}%")
        print(f"Average NJD Std          : {avg_njd_std:.2f}")
        print(f"NCC Final 95% CI         : {results['Final NCC']['95% CI']}")
        print(f"NCC Affine 95% CI        : {results['NCC Only Affine']['95% CI']}")
        print(f"NJD 95% CI               : {results['NJD']['95% CI']}")
        print(f"NJD Std 95% CI           : {results['Std dev Jacobian']['95% CI']}")