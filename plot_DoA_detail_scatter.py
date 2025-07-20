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

def plot_doa_comparison(yaml_path: str):
    config = load_yaml(yaml_path)

    base_dir = os.path.join(
        config["path"]["logdir"],
        config["path"]["expname"],
        "doa_results"
    )
    save_path = os.path.join(
        config["path"]["logdir"],
        config["path"]["expname"],
        "doa_detail_scatter.png"
    )

    pkl_paths = sorted(glob.glob(os.path.join(base_dir, "val_iter*.pkl")))

    results = []
    for path in pkl_paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if "NormMUSIC" in data and data["NormMUSIC"]["pred_vs_gt_error"]:
            errors = np.array(data["NormMUSIC"]["pred_vs_gt_error"])
            errors = errors[np.array(errors) != None]
            if len(errors) > 0:
                mean_err = np.mean(errors)
                results.append((path, mean_err))

    if not results:
        raise RuntimeError("Valid DOA results not found for NormMUSIC.")

    best_path, best_err = min(results, key=lambda x: x[1])
    last_path, last_err = results[-1]

    epoch_map = {path: i + 1 for i, (path, _) in enumerate(results)}

    def load_data(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)["NormMUSIC"]
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

    fig, axs = plt.subplots(2, 3, figsize=(21, 14))  # 拡大サイズで可視化

    for i, (path, _, label) in enumerate([
        (best_path, best_err, "Best"),
        (last_path, last_err, "Last")
    ]):
        gt, pred, true, err_pg, err_pt, err_gt, epoch = load_data(path)
        draw_subplot(axs[i, 0], gt, pred, "gt_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_gt_error: {err_pg:.2f}°")
        draw_subplot(axs[i, 1], true, pred, "true_deg", "pred_deg",
                     f"{label} (Epoch {epoch})\npred_vs_true_error: {err_pt:.2f}°")
        draw_subplot(axs[i, 2], true, gt, "true_deg", "gt_deg",
                     f"{label} (Epoch {epoch})\ngt_vs_true_error: {err_gt:.2f}°")

    fig.suptitle("DoA Results (NormMUSIC, AVR)", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOA Result Visualizer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    plot_doa_comparison(args.config)
