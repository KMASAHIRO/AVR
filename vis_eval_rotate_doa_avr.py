# plot_rotate_pred_vs_true_avr.py
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def angular_error_deg(a, b):
    """角度誤差（周期360°考慮）"""
    a = np.asarray(a) % 360.0
    b = np.asarray(b) % 360.0
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="eval_rotate_doa_avr.py の出力ディレクトリ（val_rotate_pred_with_angles.npzなどが入っている）"
    )
    ap.add_argument(
        "--save_name",
        type=str,
        default="pred_vs_true_avr.png",
        help="保存画像ファイル名（デフォルト: pred_vs_true_avr.png）"
    )
    args = ap.parse_args()

    # npzファイルを検索
    npz_paths = sorted(glob.glob(os.path.join(args.out_dir, "*.npz")))
    if not npz_paths:
        raise RuntimeError(f"No npz files found in {args.out_dir}")

    all_pred, all_true = [], []
    for p in npz_paths:
        data = np.load(p)
        if "pred_deg" in data and "true_deg" in data:
            pred = data["pred_deg"].astype(int)
            true = data["true_deg"].astype(int)
            all_pred.append(pred)
            all_true.append(true)
        else:
            print(f"⚠️ Skipped (no angles): {os.path.basename(p)}")

    if not all_pred:
        raise RuntimeError("No npz files with pred_deg / true_deg found.")

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    mae = float(angular_error_deg(pred, true).mean())

    # === 可視化 ===
    plt.figure(figsize=(7,6))
    plt.scatter(true, pred, alpha=0.5, s=14)
    plt.plot([0,360], [0,360], 'r--', linewidth=1)
    plt.xlim(0,360)
    plt.ylim(0,360)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel("True angle (deg)")
    plt.ylabel("Predicted angle (deg)")
    plt.title(f"AVR pred vs true (N={len(pred)}, MAE={mae:.2f}°)")
    plt.grid(alpha=0.3)

    save_path = os.path.join(args.out_dir, args.save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[DONE] Saved figure: {save_path}")
    print(f"        N={len(pred)}, Mean Angular Error = {mae:.3f}°")

if __name__ == "__main__":
    main()
