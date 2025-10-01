#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_partial_summaries.py

既に保存済みの results.pkl を再帰検索し、"今ある分だけ" を集計して
summary_partial_conditions.csv を出力する簡易サマライザ。

使い方例:
  python collect_partial_summaries.py --outdir path/to/outdir
  python collect_partial_summaries.py --cfg path/to/config.yaml

出力:
  <outdir>/summary_partial_conditions.csv
"""

import os
import sys
import glob
import argparse
import pickle
import math
import yaml
import pandas as pd
import numpy as np
from typing import Any, Dict, List

def summarize_from_entries(meta: Dict[str, Any], entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    # entries -> DataFrame
    rows = []
    for e in entries:
        rows.append(dict(
            seed=e.get("seed"),
            group=e.get("group"),
            pred_vs_true_error=e.get("pred_vs_true_error", float("nan")),
            pred_vs_gt_error=e.get("pred_vs_gt_error", float("nan")),
            var_circ=(e.get("pred") or {}).get("var_circ", float("nan")),
            std_circ_deg=(e.get("pred") or {}).get("std_circ_deg", float("nan")),
        ))
    if not rows:
        rows = [dict(seed=None, group=None, pred_vs_true_error=float("nan"),
                     pred_vs_gt_error=float("nan"), var_circ=float("nan"), std_circ_deg=float("nan"))]
    df = pd.DataFrame(rows)

    valid_true = df[np.isfinite(df["pred_vs_true_error"])]
    valid_gt   = df[np.isfinite(df["pred_vs_gt_error"])]

    # メタから取り出し（欠けていても落ちないように get で）
    return dict(
        condition = meta.get("condition"),
        band      = meta.get("band"),
        low_hz    = float(meta.get("low_hz", float("nan"))),
        high_hz   = float(meta.get("high_hz", float("nan"))),
        Lw_sec    = float(meta.get("Lw_sec", float("nan"))),
        Tseg_ms   = float(meta.get("Tseg_ms", float("nan"))),
        overlap   = float(meta.get("overlap", float("nan"))),
        stft_win  = meta.get("stft_win"),
        stft_nfft = int(meta.get("stft_nfft", 0)) if not pd.isna(meta.get("stft_nfft", np.nan)) else 0,
        stft_hop  = int(meta.get("stft_hop", 0)) if not pd.isna(meta.get("stft_hop", np.nan)) else 0,
        n_total       = int(len(df)),
        n_valid_true  = int(len(valid_true)),
        n_valid_gt    = int(len(valid_gt)),
        mean_pred_vs_true = float(valid_true["pred_vs_true_error"].mean()) if len(valid_true)>0 else float("nan"),
        std_pred_vs_true  = float(valid_true["pred_vs_true_error"].std(ddof=1)) if len(valid_true)>1 else float("nan"),
        mean_pred_vs_gt   = float(valid_gt["pred_vs_gt_error"].mean()) if len(valid_gt)>0 else float("nan"),
        std_pred_vs_gt    = float(valid_gt["pred_vs_gt_error"].std(ddof=1)) if len(valid_gt)>1 else float("nan"),
        mean_var_circ     = float(valid_true["var_circ"].mean()) if len(valid_true)>0 else float("nan"),
        mean_std_circ_deg = float(valid_true["std_circ_deg"].mean()) if len(valid_true)>0 else float("nan"),
    )

def summarize_from_pkl(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    meta    = data.get("meta", {})
    entries = data.get("entries", [])
    return summarize_from_entries(meta, entries)

def resolve_outdir(args) -> str:
    if args.outdir:
        return os.path.expanduser(args.outdir)
    if args.cfg:
        with open(os.path.expanduser(args.cfg), "r") as f:
            cfg = yaml.safe_load(f)
        outdir = cfg.get("outdir")
        if not outdir:
            raise ValueError("cfg に outdir がありません。--outdir を直接指定してください。")
        return os.path.expanduser(outdir)
    raise ValueError("--outdir か --cfg のどちらかを指定してください。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, help="run_bp_doa_grid.py の outdir ルート")
    ap.add_argument("--cfg", type=str, help="run_bp_doa_grid.py の YAML コンフィグ（outdir を取得）")
    args = ap.parse_args()

    outdir = resolve_outdir(args)
    if not os.path.isdir(outdir):
        print(f"[ERR] outdir が存在しません: {outdir}", file=sys.stderr)
        sys.exit(1)

    # 既存 pkl を再帰検索
    pattern = os.path.join(outdir, "**", "results.pkl")
    pkl_list = sorted(glob.glob(pattern, recursive=True))
    if not pkl_list:
        print(f"[INFO] results.pkl が見つかりませんでした: {pattern}")
        sys.exit(0)

    print(f"[INFO] 見つかった results.pkl: {len(pkl_list)} 件")
    rows: List[Dict[str, Any]] = []
    n_ok = 0
    n_ng = 0
    for p in pkl_list:
        try:
            summary = summarize_from_pkl(p)
            rows.append(summary)
            n_ok += 1
        except Exception as e:
            print(f"[WARN] 読み込み失敗: {p} -> {repr(e)}", file=sys.stderr)
            n_ng += 1

    if not rows:
        print("[INFO] 有効なサマリがありません。")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # pred_vs_gt の平均 → pred_vs_true の平均で昇順、NaN は末尾
    df_sorted = df.sort_values(
        by=["mean_pred_vs_gt", "mean_pred_vs_true"],
        ascending=[True, True],
        na_position="last"
    )

    out_csv = os.path.join(outdir, "summary_partial_conditions.csv")
    df_sorted.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DONE] 部分集計を出力しました: {out_csv}")
    print(f"       成功 {n_ok} / 失敗 {n_ng}")

if __name__ == "__main__":
    main()
