#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whitenoise_frame_eval_waveformlevel.py

--root 以下のすべての results.pkl を再帰探索し、各pklディレクトリごとに:

1) 各 entry (=1波形) について、フレーム系列角度から
   - 円平均: pred_mean_deg, gt_mean_deg
   - 円中央値: pred_median_deg, gt_median_deg
   を計算（空系列は NaN）

2) 波形ごとの代表角（平均・中央値それぞれ）を 1サンプルとして扱い、誤差を集計:
   - |gt - true|, |pred - true|, |pred - gt|
   （角度差は [-180,180) にラップ→abs）

3) 上記誤差の平均値（mean absolute error）を print

4) 散布図を保存（1行3列）
   - mean版: scatter_wave_all.png
   - median版: scatter_wave_all_median.png

※ ヒストは作らない。
"""

import os
import pickle
from glob import glob
from typing import Dict, Any, Sequence, Callable

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------- 基本ユーティリティ ----------

def wrap_deg_signed(d: np.ndarray) -> np.ndarray:
    """角度差(度)を [-180, 180) にラップ"""
    return (d + 180.0) % 360.0 - 180.0

def to_array(x, dtype=float):
    if x is None:
        return np.array([], dtype=dtype)
    return np.asarray(x, dtype=dtype)

def _extract_raw_from_entry(e: Dict[str, Any]):
    """
    entry e から pred/gt のフレーム系列と true_deg を抜き出す。
    角度系列は [0,360) に正規化。
    """
    true_deg = float(e.get("true_deg", 0.0)) % 360.0

    pred_dict = e.get("pred", {}) if isinstance(e.get("pred", {}), dict) else {}
    pred_angles = to_array(pred_dict.get("angles_deg", []), float) % 360.0

    gt_dict = e.get("gt", {}) if isinstance(e.get("gt", {}), dict) else {}
    gt_angles = to_array(gt_dict.get("angles_deg", []), float) % 360.0

    return pred_angles, gt_angles, true_deg


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """
    角度系列[deg]を円平均して [0,360) に戻す。
    空配列なら np.nan。
    """
    if angles_deg.size == 0:
        return np.nan
    rad = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    mean_rad = np.arctan2(mean_sin, mean_cos)  # [-pi, pi]
    mean_deg = np.rad2deg(mean_rad)            # [-180,180)
    return mean_deg % 360.0


def circular_median_deg(angles_deg: np.ndarray) -> float:
    """
    角度系列[deg]の「円中央値」（circular median）を返す。
    定義: 角度aに対して sum(|wrap(θ_i - a)|) が最小となる a を採用。
    解はデータ点の中に少なくとも一つ存在するため、各θ_iを候補に全探索。
    空配列なら np.nan。
    """
    if angles_deg.size == 0:
        return np.nan

    # NaN除去
    vals = angles_deg[np.isfinite(angles_deg)]
    if vals.size == 0:
        return np.nan

    candidates = vals % 360.0
    best_a = candidates[0]
    best_cost = np.inf

    for a in candidates:
        # [-180,180) にラップした絶対差の総和
        diffs = wrap_deg_signed(vals - a)
        cost = np.sum(np.abs(diffs))
        if cost < best_cost:
            best_cost = cost
            best_a = a

    return float(best_a % 360.0)


# ---------- 波形(=entry)単位での代表角度と誤差をまとめる ----------

def collect_waveform_level_data(
    entries: Sequence[dict],
    reducer: str = "mean",
):
    """
    各entryを1サンプルとして代表角を算出（mean または median）し、
      - pred_rep_deg
      - gt_rep_deg
      - true_deg
      - abs_err_* (gt-true, pred-true, pred-gt)
    を返す。

    Returns
    -------
    true_list      : np.ndarray (Nwave,)
    gt_rep_list    : np.ndarray (Nwave,) (NaNありうる)
    pred_rep_list  : np.ndarray (Nwave,) (NaNありうる)
    abs_gt_true    : np.ndarray (|gt-true| per wave, valid only where gt exists)
    abs_pred_true  : np.ndarray (|pred-true| per wave, valid only where pred exists)
    abs_pred_gt    : np.ndarray (|pred-gt| per wave, valid only where both exist)
    """
    if reducer not in ("mean", "median"):
        raise ValueError("reducer must be 'mean' or 'median'")

    reducer_fn: Callable[[np.ndarray], float] = (
        circular_mean_deg if reducer == "mean" else circular_median_deg
    )

    true_list = []
    gt_rep_list = []
    pred_rep_list = []

    abs_gt_true_list = []
    abs_pred_true_list = []
    abs_pred_gt_list = []

    for e in entries:
        pred_angles, gt_angles, true_deg = _extract_raw_from_entry(e)

        pred_rep = reducer_fn(pred_angles)
        gt_rep   = reducer_fn(gt_angles)

        true_list.append(true_deg)
        gt_rep_list.append(gt_rep)
        pred_rep_list.append(pred_rep)

        # gt vs true
        if not np.isnan(gt_rep):
            diff_gt_true = wrap_deg_signed(np.array([gt_rep - true_deg], dtype=float))
            abs_gt_true_list.append(np.abs(diff_gt_true))

        # pred vs true
        if not np.isnan(pred_rep):
            diff_pred_true = wrap_deg_signed(np.array([pred_rep - true_deg], dtype=float))
            abs_pred_true_list.append(np.abs(diff_pred_true))

        # pred vs gt
        if (not np.isnan(pred_rep)) and (not np.isnan(gt_rep)):
            diff_pred_gt = wrap_deg_signed(np.array([pred_rep - gt_rep], dtype=float))
            abs_pred_gt_list.append(np.abs(diff_pred_gt))

    def _concat(list_of_arrays):
        if len(list_of_arrays) == 0:
            return np.array([], dtype=float)
        return np.concatenate(list_of_arrays)

    return (
        np.asarray(true_list, dtype=float),
        np.asarray(gt_rep_list, dtype=float),
        np.asarray(pred_rep_list, dtype=float),
        _concat(abs_gt_true_list),
        _concat(abs_pred_true_list),
        _concat(abs_pred_gt_list),
    )


def compute_error_stats_from_arrays(abs_gt_true, abs_pred_true, abs_pred_gt):
    """
    平均絶対誤差を計算。データがなければnp.nan。
    """
    def _mean_or_nan(a):
        return float(np.nanmean(a)) if a.size > 0 else np.nan

    return {
        "mean_abs_err_gt_true":   _mean_or_nan(abs_gt_true),
        "mean_abs_err_pred_true": _mean_or_nan(abs_pred_true),
        "mean_abs_err_pred_gt":   _mean_or_nan(abs_pred_gt),
    }


# ---------- 可視化（波形単位） ----------

def plot_waveformlevel_scatter(
    out_png: str,
    true_list: np.ndarray,
    gt_rep_list: np.ndarray,
    pred_rep_list: np.ndarray,
    tag: str = "mean",
):
    """
    1波形=1点の散布図を1行3列にまとめて保存:
      (1) true vs gt_rep
      (2) true vs pred_rep
      (3) gt_rep vs pred_rep
    tag: "mean" or "median" （タイトルに反映するだけ）
    """
    if (
        true_list.size == 0 and
        gt_rep_list.size == 0 and
        pred_rep_list.size == 0
    ):
        print(f"[SKIP] no waveform scatter data -> {out_png}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.ravel()

    def scatter_plot(ax, x, y, xlabel, ylabel, title):
        mask = (~np.isnan(x)) & (~np.isnan(y))
        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, "no data", ha='center', va='center',
                    fontsize=12, color='gray')
        else:
            ax.scatter(x[mask] % 360.0, y[mask] % 360.0, s=25, alpha=0.7)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    cap = "mean" if tag == "mean" else "median"

    scatter_plot(
        axes[0],
        true_list,
        gt_rep_list,
        "true angle [deg]",
        f"gt {cap} angle [deg]",
        f"gt({cap} over time) vs true"
    )

    scatter_plot(
        axes[1],
        true_list,
        pred_rep_list,
        "true angle [deg]",
        f"pred {cap} angle [deg]",
        f"pred({cap} over time) vs true"
    )

    scatter_plot(
        axes[2],
        gt_rep_list,
        pred_rep_list,
        f"gt {cap} angle [deg]",
        f"pred {cap} angle [deg]",
        f"pred({cap}) vs gt({cap})"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close()
    print("[OK] saved:", out_png)


# ---------- メイン ----------

def _run_once_for_reducer(pkl_path: str, entries, reducer: str):
    """
    reducer = 'mean' or 'median' について集計・プロットを実行。
    """
    (
        true_list,
        gt_rep_list,
        pred_rep_list,
        abs_gt_true,
        abs_pred_true,
        abs_pred_gt,
    ) = collect_waveform_level_data(entries, reducer=reducer)

    stats = compute_error_stats_from_arrays(
        abs_gt_true,
        abs_pred_true,
        abs_pred_gt,
    )

    exp_dir = os.path.dirname(pkl_path)
    cap = "mean" if reducer == "mean" else "median"

    # 数値をprint
    print(
        f"[WAVE-LEVEL STATS - {cap}] {exp_dir}\n"
        f"  gt-true  (mean|Δ|, {cap}-over-time)   = {stats['mean_abs_err_gt_true']:.3f} deg\n"
        f"  pred-true(mean|Δ|, {cap}-over-time)   = {stats['mean_abs_err_pred_true']:.3f} deg\n"
        f"  pred-gt  (mean|Δ|, {cap}-over-time)   = {stats['mean_abs_err_pred_gt']:.3f} deg\n"
    )

    # 散布図
    out_scatter = os.path.join(
        exp_dir,
        "scatter_wave_all.png" if reducer == "mean" else "scatter_wave_all_median.png"
    )
    plot_waveformlevel_scatter(
        out_scatter,
        true_list,
        gt_rep_list,
        pred_rep_list,
        tag=reducer,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="実験ルート (例: ~/avr_tuning_logs/real_exp/whitenoise_long/exp1)")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    pkl_files = sorted(glob(os.path.join(root, "**", "results.pkl"), recursive=True))
    if not pkl_files:
        print("[ERROR] No results.pkl found under", root)
        return

    print(f"[INFO] Found {len(pkl_files)} results.pkl files")

    for pkl_path in tqdm(pkl_files, desc="WaveformLevelEval"):
        # --- load
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {pkl_path}: {e}")
            continue

        entries = data.get("entries", [])

        # === 1) 円「平均」版（後方互換: 既存の出力ファイル名を維持） ===
        _run_once_for_reducer(pkl_path, entries, reducer="mean")

        # === 2) 円「中央値」版（scatterは *_median.png） ===
        _run_once_for_reducer(pkl_path, entries, reducer="median")


if __name__ == "__main__":
    main()
