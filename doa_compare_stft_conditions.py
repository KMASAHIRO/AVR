# plot_doa_by_stft_conditions_trialwise.py
import os
import re
import glob
import pickle
import argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ====== util ======
def _load_iter_mean_from_pkl(p: str, algo: str, key: str) -> Optional[Tuple[int, float]]:
    """val_iterN.pkl -> (iter, mean_err)"""
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        errs = data.get(algo, {}).get(key, [])
        errs = [e for e in errs if e is not None]
        if not errs:
            return None
        mean_err = float(np.mean(errs))
        m = re.search(r"val_iter(\d+)\.pkl$", os.path.basename(p))
        it = int(m.group(1)) if m else -1
        return it, mean_err
    except Exception:
        return None

def _load_series_for_condition(cond_dir: str, algo: str, key: str) -> Tuple[List[int], List[float]]:
    """条件ディレクトリの val_iter*.pkl を (iters, mean_errors) で返す（iter昇順）"""
    pkl_files = sorted(glob.glob(os.path.join(cond_dir, "val_iter*.pkl")))
    xs, ys = [], []
    for p in pkl_files:
        res = _load_iter_mean_from_pkl(p, algo, key)
        if res is None:
            continue
        it, me = res
        xs.append(it); ys.append(me)
    if xs:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
    return xs, ys

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ====== main plotting ======
def run_trialwise(
    base_logdir: str,
    trial_names: List[str],
    conditions: List[str],  # 例: ["doa_rect_L128_H32", "doa_hann_L256_H64", ...]
    algo_name: str = "NormMUSIC",
    error_key: str = "pred_vs_true_error",
    ylim_doa: Tuple[float, float] = (0.0, 120.0),
    combined_minplot_path: str = "avr_tuning_logs/real_exp/doa_by_stft_conditions.png",
    csv_output_path: str = "avr_tuning_logs/real_exp/doa_by_stft_conditions.csv",
):
    """
    1) trialごとに：各条件の epoch曲線を並べた1枚の図を tdir 以下に保存
       保存先: {tdir}/doa_compare_stft_conditions/_plots/epoch_curves_grid.png
    2) 全trialを処理後：条件ごとに trial軸の最小平均誤差をまとめたパネルを1枚で保存
       保存先: combined_minplot_path
    3) CSVに trial×condition の min(mean DoA error) を保存
    """
    # 収集用（条件ごとの trialベクトル）
    cond_to_trial_idx: Dict[str, List[int]] = {c: [] for c in conditions}
    cond_to_min_errors: Dict[str, List[float]] = {c: [] for c in conditions}

    # CSV蓄積
    csv_rows = []

    for t_index, tname in enumerate(trial_names, start=1):
        tdir = os.path.join(base_logdir, tname)
        base_cond_dir = os.path.join(tdir, "doa_compare_stft_conditions")
        plots_dir = base_cond_dir
        _ensure_dir(plots_dir)

        # === この trial 内で条件ごとの epoch曲線を1枚に ===
        K = len(conditions)
        ncols = min(3, K)
        nrows = (K + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols+1.5, 3.8*nrows), squeeze=False)

        for k, cond in enumerate(conditions):
            ax = axes[k // ncols][k % ncols]
            cdir = os.path.join(base_cond_dir, cond)
            xs, ys = _load_series_for_condition(cdir, algo_name, error_key)

            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5)
                local_min = float(np.min(ys))
                title = f"{cond} | min={local_min:.2f}°"
            else:
                local_min = np.nan
                title = f"{cond} | min=NaN"

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("epoch(iter)")
            ax.set_ylabel("mean DoA error (deg)")
            ax.set_ylim(*ylim_doa)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="y", alpha=0.3)

            # trial→条件の min を蓄積（全体のtrial軸パネル用 & CSV用）
            cond_to_trial_idx[cond].append(t_index)
            cond_to_min_errors[cond].append(local_min)
            csv_rows.append({
                "trial_index": t_index,
                "trial_name": tname,
                "condition": cond,
                "min_mean_error_deg": local_min,
            })

        # 余白サブプロット削除
        for i in range(K, nrows*ncols):
            fig.delaxes(axes[i // ncols][i % ncols])

        plt.suptitle(f"Trial: {tname}  (epoch curves by STFT conditions)", y=1.02)
        plt.tight_layout()
        out_png = os.path.join(plots_dir, "epoch_curves_grid.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    # === 条件ごとの trial軸パネルを1枚に ===
    _ensure_dir(combined_minplot_path)
    K = len(conditions)
    ncols = min(3, K)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols+1.5, 3.8*nrows), squeeze=False)

    for k, cond in enumerate(conditions):
        ax = axes[k // ncols][k % ncols]
        x = cond_to_trial_idx[cond]
        y = cond_to_min_errors[cond]
        ax.plot(x, y, marker="s", linewidth=1.5, label="min mean DoA err")
        ax.set_xlabel("Trial Index")
        ax.set_ylabel("min mean DoA error (deg)")
        ax.set_ylim(*ylim_doa)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        gmin = float(np.nanmin(y)) if np.any(~np.isnan(y)) else np.nan
        ax.set_title(f"{cond} | min={gmin:.2f}°" if not np.isnan(gmin) else f"{cond} | min=NaN")
        ax.legend(loc="upper right", fontsize=8)

    for i in range(K, nrows*ncols):
        fig.delaxes(axes[i // ncols][i % ncols])

    plt.suptitle("DoA min(mean) across trials per STFT condition", y=1.02)
    plt.tight_layout()
    plt.savefig(combined_minplot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === CSV保存 ===
    _ensure_dir(csv_output_path)
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_output_path, index=False)

# ====== CLI ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial-wise STFT-condition DoA plots (no losses).")
    parser.add_argument("--logdir", type=str, default="logs/real_exp",
                        help="Base dir that contains each trial dir")
    parser.add_argument("--trial_begin", type=int, default=1)
    parser.add_argument("--trial_end", type=int, default=112)
    parser.add_argument("--trial_fmt", type=str, default="Real_exp_param_{i}_1")
    parser.add_argument("--conditions", type=str, nargs="+", required=True,
                        help='Condition subdirs under {tdir}/doa_compare_stft_conditions/, '
                             'e.g., "doa_rect_L128_H32" "doa_hann_L256_H64" ...')
    parser.add_argument("--algo_name", type=str, default="NormMUSIC")
    parser.add_argument("--error_key", type=str, default="pred_vs_true_error")
    parser.add_argument("--ylim_doa", type=float, nargs=2, default=(0.0, 120.0))
    parser.add_argument("--combined_minplot_path", type=str,
                        default="avr_tuning_logs/real_exp/doa_compare_stft_conditions.png")
    parser.add_argument("--csv_output", type=str,
                        default="avr_tuning_logs/real_exp/doa_compare_stft_conditions.csv")
    args = parser.parse_args()

    trial_names = [args.trial_fmt.format(i=i) for i in range(args.trial_begin, args.trial_end + 1)]

    run_trialwise(
        base_logdir=args.logdir,
        trial_names=trial_names,
        conditions=args.conditions,
        algo_name=args.algo_name,
        error_key=args.error_key,
        ylim_doa=tuple(args.ylim_doa),
        combined_minplot_path=args.combined_minplot_path,
        csv_output_path=args.csv_output,
    )
