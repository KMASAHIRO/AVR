import os
import glob
import pickle
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def angular_error_deg(a, b):
    return np.minimum(np.abs(a - b), 360 - np.abs(a - b))

def plot_das_comparison(yaml_path: str):
    config = load_yaml(yaml_path)

    base_dir = os.path.join(
        config["path"]["logdir"],
        config["path"]["expname"],
        "beamform_results"
    )
    save_path = os.path.join(
        config["path"]["logdir"],
        config["path"]["expname"],
        "das_detail_scatter.png"
    )

    pkl_paths = sorted(glob.glob(os.path.join(base_dir, "val_iter*.pkl")))

    results_soft = []
    results_argmax = []

    for path in pkl_paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # soft-argmax
        errors_soft = [e for e in data["NormDAS_soft-argmax"]["pred_vs_gt_error"] if e is not None]
        if len(errors_soft) > 0:
            results_soft.append((path, np.mean(errors_soft)))

        # argmax
        errors_arg = [e for e in data["NormDAS_argmax"]["pred_vs_gt_error"] if e is not None]
        if len(errors_arg) > 0:
            results_argmax.append((path, np.mean(errors_arg)))


    if not results_soft or not results_argmax:
        raise RuntimeError("Valid DAS results not found for soft-argmax or argmax.")

    best_soft_path, best_soft_err = min(results_soft, key=lambda x: x[1])
    last_soft_path, last_soft_err = results_soft[-1]

    best_arg_path, best_arg_err = min(results_argmax, key=lambda x: x[1])
    last_arg_path, last_arg_err = results_argmax[-1]

    epoch_map_soft = {path: i + 1 for i, (path, _) in enumerate(results_soft)}
    epoch_map_arg = {path: i + 1 for i, (path, _) in enumerate(results_argmax)}

    def load_data(path, method_name, epoch_map):
        with open(path, 'rb') as f:
            d = pickle.load(f)[method_name]
        gt_deg = np.array(d["gt_deg"])
        pred_deg = np.array(d["pred_deg"])
        true_deg = np.array(d["true_deg"])
        err_pred_gt = np.mean([e for e in d["pred_vs_gt_error"] if e is not None])
        err_pred_true = np.mean([e for e in d["pred_vs_true_error"] if e is not None])
        err_gt_true = np.mean([e for e in d["gt_vs_true_error"] if e is not None])
        epoch = epoch_map[path]
        return gt_deg, pred_deg, true_deg, err_pred_gt, err_pred_true, err_gt_true, epoch

    def draw_subplot(ax, x, y, xlabel, ylabel, title):
        ax.scatter(x, y, alpha=0.5)
        ax.plot([0, 360], [0, 360], 'r--')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    fig, axs = plt.subplots(4, 3, figsize=(21, 28))  # 縦4行に拡大

    # Soft-argmax (上2行)
    for i, (path, label, epoch_map) in enumerate([
        (best_soft_path, "Soft - Best", epoch_map_soft),
        (last_soft_path, "Soft - Last", epoch_map_soft)
    ]):
        gt, pred, true, err_pg, err_pt, err_gt, epoch = load_data(path, "NormDAS_soft-argmax", epoch_map)
        draw_subplot(axs[i, 0], gt, pred, "gt_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_gt_error: {err_pg:.2f}°")
        draw_subplot(axs[i, 1], true, pred, "true_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_true_error: {err_pt:.2f}°")
        draw_subplot(axs[i, 2], true, gt, "true_deg", "gt_deg",
                     f"{label} (Epoch {epoch})\ngt_vs_true_error: {err_gt:.2f}°")

    # Argmax (下2行)
    for j, (path, label, epoch_map) in enumerate([
        (best_arg_path, "Argmax - Best", epoch_map_arg),
        (last_arg_path, "Argmax - Last", epoch_map_arg)
    ]):
        gt, pred, true, err_pg, err_pt, err_gt, epoch = load_data(path, "NormDAS_argmax", epoch_map)
        draw_subplot(axs[j+2, 0], gt, pred, "gt_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_gt_error: {err_pg:.2f}°")
        draw_subplot(axs[j+2, 1], true, pred, "true_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_true_error: {err_pt:.2f}°")
        draw_subplot(axs[j+2, 2], true, gt, "true_deg", "gt_deg",
                     f"{label} (Epoch {epoch})\ngt_vs_true_error: {err_gt:.2f}°")

    fig.suptitle("DAS Results (Soft-argmax & Argmax)", fontsize=26)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAS Result Visualizer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    plot_das_comparison(args.config)
