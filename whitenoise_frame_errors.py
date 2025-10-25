#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whitenoise_frame_plots.py

機能:
  指定root配下の全 results.pkl を再帰探索し、以下2種のグリッド画像を出力
  (各 pkl と同じフォルダに保存)

  1) frame_errors_grid.png
     - 横軸: STFT 窓中心フレーム index
     - 縦軸: (pred角度 - 真値) [deg, -180..180]
     - 各サブプロットは1グループ (全18 = 3x6想定)

  2) angles_pred_gt_grid.png
     - 横軸: STFT 窓中心フレーム index
     - 縦軸: 角度 [deg, 0..360)
     - pred（青）と gt（橙; あれば）を同一座標に散布
     - 真値 true_deg に赤水平線を表示
       ※ gt 側に frame 系列 (angles_deg / centers) が保存されていない pkl は、
          pred のみ表示し、凡例に "gt: n/a" と表記

使い方:
  python whitenoise_frame_plots.py --root ~/avr_tuning_logs/real_exp/whitenoise_long/exp1
"""

import os
import math
import pickle
from glob import glob
from typing import Tuple, Sequence

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

def _title_for_subplot(e) -> str:
    g = e.get('group', '?')
    td = float(e.get('true_deg', float('nan')))
    return f"G{int(g):02d} (true={td:.1f}°)"


# ---------------------- 1) 誤差グリッド ----------------------

def plot_errors_grid(pkl_path: str, rows: int = 3, cols: int = 6,
                     ylim: Tuple[float, float] = (-60, 60), dpi: int = 250) -> None:
    """pred の (角度 - 真値) を散布で表示"""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {pkl_path}: {e}")
        return

    entries = data.get("entries", [])
    if not entries:
        print(f"[SKIP] no entries in {pkl_path}")
        return

    entries_sorted = sorted(entries, key=lambda e: int(e.get("group", 9999)))
    R, C = rows, cols
    fig, axes = plt.subplots(R, C, figsize=(4.6*C, 2.9*R), sharex=False, sharey=True)
    axes = axes.ravel()
    last_ax_idx = -1

    for i, e in enumerate(entries_sorted):
        last_ax_idx = i
        ax = axes[i]

        true_deg = float(e.get("true_deg", 0.0))
        pred_angles = to_array(e.get("pred", {}).get("angles_deg", []), float)
        pred_centers = to_array(e.get("pred", {}).get("centers", []), int)

        if pred_angles.size == 0 or pred_centers.size == 0:
            ax.set_title(_title_for_subplot(e) + "\n[no pred frames]", fontsize=9)
            ax.axhline(0.0, color="red", lw=1.2, alpha=0.85)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Window center (STFT frame)")
            if i % C == 0:
                ax.set_ylabel("Angle error [deg]")
            continue

        err = wrap_deg_signed(pred_angles - true_deg)
        ax.scatter(pred_centers, err, s=6, c='royalblue', alpha=0.65, label="pred error")
        ax.axhline(0.0, color="red", lw=1.2, alpha=0.9)
        ax.set_title(_title_for_subplot(e), fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Window center (STFT frame)")
        if i % C == 0:
            ax.set_ylabel("Angle error [deg]")

    # 余白の空き軸は削除
    for j in range(last_ax_idx + 1, R * C):
        fig.delaxes(axes[j])

    # 全体タイトル
    meta = data.get("meta", {})
    cond = meta.get("condition", "")
    stft = f"L{meta.get('stft_nfft','?')}_H{meta.get('stft_hop','?')}_{meta.get('stft_win','?')}"
    Tuse = meta.get("Tuse", None)
    fig.suptitle(f"{cond} | STFT {stft}" + (f" | T_use={Tuse}" if Tuse is not None else "") +
                 "  (pred - true)", fontsize=12, y=1.02)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(pkl_path), "frame_errors_grid.png")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", out_png)


# ---------------------- 2) 角度（pred / gt）グリッド ----------------------

def plot_angles_grid(pkl_path: str, rows: int = 3, cols: int = 6,
                     ylim: Tuple[float, float] = (0.0, 360.0), dpi: int = 250) -> None:
    """pred と gt（あれば）の角度自体を散布で表示。true_deg は赤水平線。"""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {pkl_path}: {e}")
        return

    entries = data.get("entries", [])
    if not entries:
        print(f"[SKIP] no entries in {pkl_path}")
        return

    entries_sorted = sorted(entries, key=lambda e: int(e.get("group", 9999)))
    R, C = rows, cols
    fig, axes = plt.subplots(R, C, figsize=(4.6*C, 2.9*R), sharex=False, sharey=True)
    axes = axes.ravel()
    last_ax_idx = -1

    for i, e in enumerate(entries_sorted):
        last_ax_idx = i
        ax = axes[i]

        true_deg = float(e.get("true_deg", 0.0))

        # pred
        pred_angles = to_array(e.get("pred", {}).get("angles_deg", []), float) % 360.0
        pred_centers = to_array(e.get("pred", {}).get("centers", []), int)

        # gt（あれば; ない場合はスキップ）
        gt_dict = e.get("gt", {}) if isinstance(e.get("gt", {}), dict) else {}
        gt_angles = to_array(gt_dict.get("angles_deg", []), float) % 360.0
        gt_centers = to_array(gt_dict.get("centers", []), int)

        has_pred = (pred_angles.size > 0 and pred_centers.size > 0)
        has_gt = (gt_angles.size > 0 and gt_centers.size > 0)

        if not has_pred and not has_gt:
            ax.set_title(_title_for_subplot(e) + "\n[pred/gt: no frames]", fontsize=9)
            ax.axhline(true_deg % 360.0, color="red", lw=1.0, alpha=0.9, linestyle='--')
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Window center (STFT frame)")
            if i % C == 0:
                ax.set_ylabel("Angle [deg]")
            continue

        if has_pred:
            ax.scatter(pred_centers, pred_angles, s=6, c='royalblue', alpha=0.65, label="pred")
        else:
            ax.text(0.02, 0.85, "pred: n/a", transform=ax.transAxes, fontsize=8, color='royalblue')

        if has_gt:
            ax.scatter(gt_centers, gt_angles, s=8, marker='^', c='darkorange', alpha=0.7, label="gt")
        else:
            ax.text(0.02, 0.75, "gt: n/a", transform=ax.transAxes, fontsize=8, color='darkorange')

        ax.axhline(true_deg % 360.0, color="red", lw=1.2, alpha=0.9, linestyle='-')
        ax.set_title(_title_for_subplot(e), fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Window center (STFT frame)")
        if i % C == 0:
            ax.set_ylabel("Angle [deg]")

        # 軽い凡例（最左上だけ出す）
        if i == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    # 余白の空き軸は削除
    for j in range(last_ax_idx + 1, R * C):
        fig.delaxes(axes[j])

    # 全体タイトル
    meta = data.get("meta", {})
    cond = meta.get("condition", "")
    stft = f"L{meta.get('stft_nfft','?')}_H{meta.get('stft_hop','?')}_{meta.get('stft_win','?')}"
    Tuse = meta.get("Tuse", None)
    fig.suptitle(f"{cond} | STFT {stft}" + (f" | T_use={Tuse}" if Tuse is not None else "") +
                 "  (angles: pred & gt)", fontsize=12, y=1.02)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(pkl_path), "angles_pred_gt_grid.png")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", out_png)


# ---------------------- メイン処理 ----------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="実験ルート (例: ~/avr_tuning_logs/real_exp/whitenoise_long/exp1)")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--err_ylim", type=float, nargs=2, default=[-60, 60])
    ap.add_argument("--ang_ylim", type=float, nargs=2, default=[0, 360])
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    pkl_files = sorted(glob(os.path.join(root, "**", "results.pkl"), recursive=True))
    if not pkl_files:
        print("[ERROR] No results.pkl found under", root)
        return

    print(f"[INFO] Found {len(pkl_files)} results.pkl files")
    for pkl_path in tqdm(pkl_files, desc="Plotting"):
        plot_errors_grid(pkl_path, rows=args.rows, cols=args.cols,
                         ylim=tuple(args.err_ylim), dpi=args.dpi)
        plot_angles_grid(pkl_path, rows=args.rows, cols=args.cols,
                         ylim=tuple(args.ang_ylim), dpi=args.dpi)

if __name__ == "__main__":
    main()
