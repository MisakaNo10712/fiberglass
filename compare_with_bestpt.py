import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import existing model definitions
from directrun import read_fiber_txt, Normalizer, build_plate_basis, PlateBasisNet


@torch.no_grad()
def compare_prediction(
    run_dir: str = "runs/plate_basis",
    data_file: str = "data/merged_fibers_output_z_p0p000.txt",
    ckpt_name: str = "best.pt",
    save_dir: str = None,
):
    """
    Use best.pt model to predict data from data folder and compare with ground truth
    
    Args:
        run_dir: Model save directory
        data_file: Test data file path
        ckpt_name: Model checkpoint name
        save_dir: Results save directory
    """
    if save_dir is None:
        save_dir = run_dir

    ckpt_path = os.path.join(run_dir, ckpt_name)
    cfg_path = os.path.join(run_dir, "config.json")
    norm_path = os.path.join(run_dir, "normalizers.npz")

    # Check if files exist
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    assert os.path.exists(cfg_path), f"Missing config: {cfg_path}"
    assert os.path.exists(norm_path), f"Missing normalizers: {norm_path}"
    assert os.path.exists(data_file), f"Missing data file: {data_file}"

    # Load config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    d_model = int(cfg.get("d_model", 128))
    dropout = float(cfg.get("dropout", 0.1))

    # Load normalizer
    nz = np.load(norm_path)
    strain_norm = Normalizer(nz["strain_mean"], nz["strain_std"])
    disp_norm = Normalizer(nz["disp_mean"], nz["disp_std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    W = float(ckpt["W"])
    L = float(ckpt["L"])
    M = int(ckpt["M"])
    N = int(ckpt["N"])
    clamp_y = bool(ckpt["clamp_y"])

    print(f"[INFO] Plate parameters: W={W:.4f}, L={L:.4f}, M={M}, N={N}, clamp_y={clamp_y}")

    # Load test data
    print(f"[INFO] Loading data file: {data_file}")
    d = read_fiber_txt(data_file)
    coords = d["coords"]  # (P,3)
    strain = d["strain"]  # (P,1)
    disp_true = d["disp"]  # (P,3) - ground truth displacement

    # Prepare input features
    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)
    coord_feat_sensor = np.stack([x / W, y / L], axis=1).astype(np.float32)  # (P,2)

    # Convert to tensor
    strain_t = torch.from_numpy(strain).unsqueeze(0).to(device)  # (1,P,1)
    strain_n = strain_norm.encode_t(strain_t)  # (1,P,1)
    coord_t = torch.from_numpy(coord_feat_sensor).unsqueeze(0).to(device)  # (1,P,2)
    feats = torch.cat([coord_t, strain_n], dim=-1)  # (1,P,3)

    # Build basis for sensor points
    coords_xy_t = torch.from_numpy(np.stack([x, y], axis=1)).float().to(device)
    Phi_sensor, meta = build_plate_basis(coords_xy_t, W=W, L=L, M=M, N=N, clamp_y=clamp_y)

    # Build model
    model = PlateBasisNet(
        Phi=Phi_sensor,
        meta=meta,
        in_feature_dim=3,
        d_model=d_model,
        out_channels=3,
        dropout=dropout,
    ).to(device)
    model.eval()

    # Load weights (skip Phi-related parameters)
    sd = ckpt["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("Phi_")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded model weights - missing: {len(missing)}, unexpected: {len(unexpected)}")

    # Predict
    pred_n, coeffs = model(feats)  # (1,P,3)
    pred = disp_norm.decode_t(pred_n)[0]  # (P,3)
    pred = pred.detach().cpu().numpy()

    # Calculate errors
    error = pred - disp_true
    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt((error ** 2).mean(axis=0))
    max_error = np.abs(error).max(axis=0)

    print("\n" + "="*60)
    print("Prediction Statistics:")
    print("="*60)
    print(f"{'Component':<10} {'MAE':<15} {'RMSE':<15} {'Max Error':<15}")
    print("-"*60)
    for i, name in enumerate(['u', 'v', 'w']):
        print(f"{name:<10} {mae[i]:<15.6e} {rmse[i]:<15.6e} {max_error[i]:<15.6e}")
    print("="*60)

    # Calculate relative errors
    for i, name in enumerate(['u', 'v', 'w']):
        true_range = disp_true[:, i].max() - disp_true[:, i].min()
        if true_range > 1e-10:
            rel_rmse = rmse[i] / true_range * 100
            print(f"{name} Relative RMSE: {rel_rmse:.2f}%")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Save numerical results
    result_file = os.path.join(save_dir, "comparison_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Data file: {data_file}\n")
        f.write(f"Model: {ckpt_path}\n\n")
        f.write("Error Statistics:\n")
        f.write(f"{'Component':<10} {'MAE':<15} {'RMSE':<15} {'Max Error':<15}\n")
        for i, name in enumerate(['u', 'v', 'w']):
            f.write(f"{name:<10} {mae[i]:<15.6e} {rmse[i]:<15.6e} {max_error[i]:<15.6e}\n")
    print(f"\n[INFO] Results saved to: {result_file}")

    # Visualize comparison
    visualize_comparison(coords, disp_true, pred, error, save_dir, data_file)

    return pred, disp_true, error


def visualize_comparison(coords, disp_true, disp_pred, error, save_dir, data_file):
    """
    Visualize comparison of ground truth, prediction and error
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Create main figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    components = ['u', 'v', 'w']
    
    for i, comp in enumerate(components):
        true_vals = disp_true[:, i]
        pred_vals = disp_pred[:, i]
        err_vals = error[:, i]
        
        # Ground truth
        ax1 = fig.add_subplot(gs[i, 0])
        sc1 = ax1.scatter(x, y, c=true_vals, cmap='viridis', s=10)
        ax1.set_title(f'{comp} - Ground Truth', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(sc1, ax=ax1)
        ax1.set_aspect('equal')
        
        # Prediction
        ax2 = fig.add_subplot(gs[i, 1])
        sc2 = ax2.scatter(x, y, c=pred_vals, cmap='viridis', s=10)
        ax2.set_title(f'{comp} - Prediction', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(sc2, ax=ax2)
        ax2.set_aspect('equal')
        
        # Error
        ax3 = fig.add_subplot(gs[i, 2])
        sc3 = ax3.scatter(x, y, c=err_vals, cmap='RdBu_r', s=10)
        ax3.set_title(f'{comp} - Error', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(sc3, ax=ax3)
        ax3.set_aspect('equal')
        
        # Scatter plot comparison
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.scatter(true_vals, pred_vals, alpha=0.5, s=5)
        
        # Calculate R²
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Plot ideal line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        
        ax4.set_xlabel(f'{comp} Ground Truth')
        ax4.set_ylabel(f'{comp} Prediction')
        ax4.set_title(f'{comp} - Correlation (R²={r2:.4f})', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
    
    # Add overall title
    data_name = os.path.basename(data_file)
    fig.suptitle(f'Prediction Comparison - {data_name}', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    save_path = os.path.join(save_dir, 'comparison_visualization.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {save_path}")
    
    # Create error distribution plot
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, comp in enumerate(components):
        err_vals = error[:, i]
        axes[i].hist(err_vals, bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel(f'{comp} Error')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{comp} Error Distribution\nMean={err_vals.mean():.2e}, Std={err_vals.std():.2e}')
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    error_dist_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(error_dist_path, dpi=200, bbox_inches='tight')
    print(f"[INFO] Error distribution saved to: {error_dist_path}")
    
    plt.close('all')


if __name__ == "__main__":
    # Modify here to test different data files
    import sys
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Default: use first data file
        data_file = "data/merged_fibers_output_z_p0p000.txt"
    
    print(f"\n{'='*60}")
    print(f"Prediction and Comparison using best.pt model")
    print(f"{'='*60}\n")
    
    compare_prediction(
        run_dir="runs/plate_basis",
        data_file=data_file,
        ckpt_name="best.pt",
        save_dir="runs/plate_basis/comparison"
    )
    
    print("\nDone!")
