#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whitenoise_frame_eval_all.py

--root 以下のすべての results.pkl を再帰探索し、各pklディレクトリごとに:

1. 誤差統計のprint
   - mean|gt-true|
   - mean|pred-true|
   - mean|pred-gt|（共通frameのみ）

2. 散布図 (scatter_all.png, 1行3列)
   - (1) x=true, y=gt
   - (2) x=true, y=pred
   - (3) x=gt,   y=pred（同一frameのみ）

3. エラーヒスト (error_hist.png, 1行3列)
   - |gt-true| の分布
   - |pred-true| の分布
   - |pred-gt| の分布
     ※ pred-gtは共通frameのみ

角度は [0,360) で扱う。
誤差は [-180,180) にラップしてから abs(...)。
"""

import os
import pickle
from glob import glob
from typing import Dict, Any, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------- 基本ユーティリティ ----------------------

def wrap_deg_signed(d: np.ndarray) -> np.ndarray:
    """角度差(度)を [-180, 180) にラップ"""
    return (d + 180.0) % 360.0 - 180.0


def to_array(x, dtype=float):
    if x is None:
        return np.array([], dtype=dtype)
    return np.asarray(x, dtype=dtype)


def _extract_pred_gt_arrays_from_entry(e: Dict[str, Any]):
    """
    entry e から pred / gt / true_deg を取り出す。
    角度は [0,360) に正規化。
    """
    true_deg = float(e.get("true_deg", 0.0)) % 360.0

    pred_dict = e.get("pred", {}) if isinstance(e.get("pred", {}), dict) else {}
    pred_angles = to_array(pred_dict.get("angles_deg", []), float) % 360.0
    pred_centers = to_array(pred_dict.get("centers", []), int)

    gt_dict = e.get("gt", {}) if isinstance(e.get("gt", {}), dict) else {}
    gt_angles = to_array(gt_dict.get("angles_deg", []), float) % 360.0
    gt_centers = to_array(gt_dict.get("centers", []), int)

    return pred_angles, pred_centers, gt_angles, gt_centers, true_deg


# ---------------------- データ収集 ----------------------

def collect_scatter_points(entries: Sequence[dict]):
    """
    entries全体から散布用データを収集する。

    Returns
    -------
    xs_true_gt, ys_true_gt        : x=true_deg, y=gt_angles
    xs_true_pred, ys_true_pred    : x=true_deg, y=pred_angles
    xs_gt_pred, ys_gt_pred        : x=gt_angles, y=pred_angles (共通frameのみ)
    """
    xs_true_gt, ys_true_gt = [], []
    xs_true_pred, ys_true_pred = [], []
    xs_gt_pred, ys_gt_pred = [], []

    for e in entries:
        pred_angles, pred_centers, gt_angles, gt_centers, true_deg = \
            _extract_pred_gt_arrays_from_entry(e)

        # true vs gt
        if gt_angles.size > 0:
            xs_true_gt.append(np.full_like(gt_angles, fill_value=true_deg, dtype=float))
            ys_true_gt.append(gt_angles)

        # true vs pred
        if pred_angles.size > 0:
            xs_true_pred.append(np.full_like(pred_angles, fill_value=true_deg, dtype=float))
            ys_true_pred.append(pred_angles)

        # gt vs pred (同じframeのみ)
        if (gt_angles.size > 0) and (pred_angles.size > 0):
            pred_map = {int(f): ang for f, ang in zip(pred_centers.tolist(), pred_angles.tolist())}
            gt_map   = {int(f): ang for f, ang in zip(gt_centers.tolist(), gt_angles.tolist())}
            common_frames = sorted(set(pred_map.keys()) & set(gt_map.keys()))
            if len(common_frames) > 0:
                xs_gt_pred.append(np.array([gt_map[f]   for f in common_frames], dtype=float))
                ys_gt_pred.append(np.array([pred_map[f] for f in common_frames], dtype=float))

    def _concat(list_of_arrays):
        if len(list_of_arrays) == 0:
            return np.array([], dtype=float)
        return np.concatenate(list_of_arrays)

    return (
        _concat(xs_true_gt),  _concat(ys_true_gt),
        _concat(xs_true_pred), _concat(ys_true_pred),
        _concat(xs_gt_pred),   _concat(ys_gt_pred),
    )


def collect_error_arrays(entries: Sequence[dict]):
    """
    entriesから、誤差の絶対値 [deg] をフラットに集める。
    - |gt - true|
    - |pred - true|
    - |pred - gt|（共通frameのみ）
    """

    all_abs_err_gt_true = []
    all_abs_err_pred_true = []
    all_abs_err_pred_gt = []

    for e in entries:
        pred_angles, pred_centers, gt_angles, gt_centers, true_deg = \
            _extract_pred_gt_arrays_from_entry(e)

        # gt vs true
        if gt_angles.size > 0:
            diff_gt_true = wrap_deg_signed(gt_angles - true_deg)
            all_abs_err_gt_true.append(np.abs(diff_gt_true))

        # pred vs true
        if pred_angles.size > 0:
            diff_pred_true = wrap_deg_signed(pred_angles - true_deg)
            all_abs_err_pred_true.append(np.abs(diff_pred_true))

        # pred vs gt (共通frameのみ)
        if (gt_angles.size > 0) and (pred_angles.size > 0):
            pred_map = {int(f): ang for f, ang in zip(pred_centers.tolist(), pred_angles.tolist())}
            gt_map   = {int(f): ang for f, ang in zip(gt_centers.tolist(), gt_angles.tolist())}
            common_frames = sorted(set(pred_map.keys()) & set(gt_map.keys()))
            if len(common_frames) > 0:
                pa = np.array([pred_map[f] for f in common_frames], dtype=float)
                ga = np.array([gt_map[f]   for f in common_frames], dtype=float)
                diff_pred_gt = wrap_deg_signed(pa - ga)
                all_abs_err_pred_gt.append(np.abs(diff_pred_gt))

    def _concat(list_of_arrays):
        if len(list_of_arrays) == 0:
            return np.array([], dtype=float)
        return np.concatenate(list_of_arrays)

    return (
        _concat(all_abs_err_gt_true),
        _concat(all_abs_err_pred_true),
        _concat(all_abs_err_pred_gt),
    )


def compute_error_stats(abs_err_gt_true: np.ndarray,
                        abs_err_pred_true: np.ndarray,
                        abs_err_pred_gt: np.ndarray) -> dict:
    """
    各誤差配列から平均値を計算する。
    データがなければ np.nan。
    """
    def _mean_or_nan(a: np.ndarray):
        if a.size == 0:
            return np.nan
        return float(np.nanmean(a))

    return {
        "mean_abs_err_gt_true":   _mean_or_nan(abs_err_gt_true),
        "mean_abs_err_pred_true": _mean_or_nan(abs_err_pred_true),
        "mean_abs_err_pred_gt":   _mean_or_nan(abs_err_pred_gt),
    }


# ---------------------- 可視化 ----------------------

def plot_scatter_all(
    out_png: str,
    xs_true_gt: np.ndarray, ys_true_gt: np.ndarray,
    xs_true_pred: np.ndarray, ys_true_pred: np.ndarray,
    xs_gt_pred: np.ndarray, ys_gt_pred: np.ndarray
):
    """
    3つの散布図を1行3列にまとめて scatter_all.png として保存
    """
    # 何も点がない場合はスキップ
    if (
        xs_true_gt.size == 0 and
        xs_true_pred.size == 0 and
        xs_gt_pred.size == 0
    ):
        print(f"[SKIP] no scatter data -> {out_png}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.ravel()

    def scatter_plot(ax, x, y, xlabel, ylabel, title):
        if x.size == 0 or y.size == 0:
            ax.text(0.5, 0.5, "no data", ha='center', va='center',
                    fontsize=12, color='gray')
        else:
            ax.scatter(x, y, s=10, alpha=0.6)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    scatter_plot(axes[0], xs_true_gt,  ys_true_gt,
                 "true angle [deg]", "gt angle [deg]",   "gt vs true")
    scatter_plot(axes[1], xs_true_pred, ys_true_pred,
                 "true angle [deg]", "pred angle [deg]", "pred vs true")
    scatter_plot(axes[2], xs_gt_pred,   ys_gt_pred,
                 "gt angle [deg]",   "pred angle [deg]", "pred vs gt")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close()
    print("[OK] saved:", out_png)


def plot_error_hist(
    out_png: str,
    abs_err_gt_true: np.ndarray,
    abs_err_pred_true: np.ndarray,
    abs_err_pred_gt: np.ndarray
):
    """
    3つの誤差絶対値分布を1行3列に並べてヒストグラム化して保存。
    X軸は誤差[deg]、Y軸は頻度。
    """
    if (
        abs_err_gt_true.size == 0 and
        abs_err_pred_true.size == 0 and
        abs_err_pred_gt.size == 0
    ):
        print(f"[SKIP] no error data -> {out_png}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.ravel()

    def hist_plot(ax, data, title):
        if data.size == 0:
            ax.text(0.5, 0.5, "no data", ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_xlim(0, 180)
            ax.set_xlabel("|Δangle| [deg]")
            ax.set_ylabel("count")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            return

        # 絶対誤差なので0〜180の範囲に入るはず
        ax.hist(data, bins=36, range=(0,180), alpha=0.7)
        ax.set_xlim(0, 180)
        ax.set_xlabel("|Δangle| [deg]")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    hist_plot(axes[0], abs_err_gt_true,   "abs(gt - true)")
    hist_plot(axes[1], abs_err_pred_true, "abs(pred - true)")
    hist_plot(axes[2], abs_err_pred_gt,   "abs(pred - gt)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches='tight')
    plt.close()
    print("[OK] saved:", out_png)


# ---------------------- メイン処理 ----------------------

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
    for pkl_path in tqdm(pkl_files, desc="EvalAll"):
        # --- load
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {pkl_path}: {e}")
            continue

        entries = data.get("entries", [])

        # --- 1) 散布用データ収集
        (
            xs_true_gt,  ys_true_gt,
            xs_true_pred, ys_true_pred,
            xs_gt_pred,   ys_gt_pred,
        ) = collect_scatter_points(entries)

        # --- 2) 誤差ベクトル収集
        (
            abs_err_gt_true,
            abs_err_pred_true,
            abs_err_pred_gt,
        ) = collect_error_arrays(entries)

        # --- 3) 統計値計算
        stats = compute_error_stats(abs_err_gt_true,
                                    abs_err_pred_true,
                                    abs_err_pred_gt)

        # --- 4) print（見やすい整形）
        exp_dir = os.path.dirname(pkl_path)
        print(
            f"[STATS] {exp_dir}\n"
            f"  gt-true (mean|Δ|)   = {stats['mean_abs_err_gt_true']:.3f} deg\n"
            f"  pred-true (mean|Δ|) = {stats['mean_abs_err_pred_true']:.3f} deg\n"
            f"  pred-gt (mean|Δ|)   = {stats['mean_abs_err_pred_gt']:.3f} deg\n"
        )

        # --- 5) 散布3枚まとめ
        out_scatter = os.path.join(exp_dir, "scatter_all.png")
        plot_scatter_all(
            out_scatter,
            xs_true_gt,  ys_true_gt,
            xs_true_pred, ys_true_pred,
            xs_gt_pred,   ys_gt_pred,
        )

        # --- 6) エラーヒスト3枚まとめ
        out_hist = os.path.join(exp_dir, "error_hist.png")
        plot_error_hist(
            out_hist,
            abs_err_gt_true,
            abs_err_pred_true,
            abs_err_pred_gt,
        )


if __name__ == "__main__":
    main()
