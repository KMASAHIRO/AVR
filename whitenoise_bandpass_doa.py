#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bp_doa_grid.py (resume-safe, pkl reuse, --force switch)

- 単一npz内の 18×8ch freq-IR（pred/ori）から
  * 白色雑音×IRで観測合成（各seedで共通x）
  * Butterworth(4) bandpass → filtfilt（ゼロ位相）
  * 時間フレーミング（Tseg, overlap）
  * STFT-DoA（nfft, hop_size, window ∈ {none, hann}）
  * 各フレーム角度を円平均 → サンプル代表角度
  * 誤差:
      - pred_vs_true_error: 幾何真値（tx位置）との角度差
      - pred_vs_gt_error:   同条件で ori-IRから求めた代表角度との差
- 出力
  * 条件ごと: results.pkl（生データすべて; 統計の取り直し自在）
  * まとめ: summary_all_conditions.csv（pred_vs_true平均の昇順）

依存: numpy scipy pandas pyroomacoustics pyyaml
"""

import os, math, argparse, yaml, pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal
import pyroomacoustics as pra

# -------------------- 角度ユーティリティ --------------------

def angular_error_deg(a_deg: float, b_deg: float) -> float:
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)

def circ_mean_deg(angles_deg: List[float]) -> Tuple[float, float]:
    if len(angles_deg) == 0:
        return float("nan"), 0.0
    a = np.deg2rad(np.asarray(angles_deg))
    C = float(np.cos(a).sum()); S = float(np.sin(a).sum())
    mu = (np.rad2deg(math.atan2(S, C)) + 360.0) % 360.0
    R = math.hypot(C, S) / max(len(angles_deg), 1)
    return mu, R

def circ_stats_deg(angles_deg: List[float]) -> Tuple[float, float, float]:
    mu, R = circ_mean_deg(angles_deg)
    var = 1.0 - R
    std_deg = np.rad2deg(np.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12))))) if R > 0 else float("nan")
    return mu, var, std_deg

# -------------------- 入出力 --------------------

@dataclass
class Config:
    npz: str
    which: str
    fs: int
    seeds: List[int]
    bands: List[Dict[str, Any]]
    noise_seconds: List[float]
    segments_ms: List[float]
    overlap_factors: List[float]       # 例: [0.0, 0.25, 0.5] など（seg/hop を決める）
    stft_grid: List[Dict[str, Any]]    # {nfft, hop: samples, win: "none"/"hann"}
    outdir: str
    mic_radius: float = 0.0365
    algo_name: str = "NormMUSIC"
    force: bool = False                # ← 追加：cfg 側にも反映可能に

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_npz_freq_ir(npz_path: str):
    d = np.load(os.path.expanduser(npz_path))
    H_pred = d["pred_sig"]           # (N, F)
    H_ori  = d.get("ori_sig", None)  # (N, F) or None
    pos_rx = d.get("position_rx", None)  # (N,3)
    pos_tx = d.get("position_tx", None)  # (N,3)
    return H_pred, H_ori, pos_rx, pos_tx

def groups_of_8(arr: np.ndarray) -> List[np.ndarray]:
    N = arr.shape[0]
    if N % 8 != 0:
        raise ValueError(f"N={N} is not divisible by 8.")
    return [arr[g*8:(g+1)*8] for g in range(N // 8)]

# -------------------- 合成・前処理 --------------------

def white_noise(Lw_sec: float, fs: int, seed: int) -> np.ndarray:
    Nt = int(round(Lw_sec * fs))
    rng = np.random.default_rng(seed)
    return rng.standard_normal(Nt).astype(np.float32)

def synth_observation(ir_group_freq: np.ndarray, x: np.ndarray) -> np.ndarray:
    h = np.fft.irfft(ir_group_freq, axis=1).astype(np.float32)  # (8, Nh)
    ys = [np.convolve(x, h_i, mode="full") for h_i in h]
    return np.stack(ys, axis=0).astype(np.float32)              # (8, T)

def butter_bandpass_sos(low_hz: float, high_hz: float, fs: int, order: int = 4):
    return signal.butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")

def apply_zero_phase_bp(y: np.ndarray, sos) -> np.ndarray:
    return signal.sosfiltfilt(sos, y, axis=-1)

def seg_hop_samples(fs: int, Tseg_ms: float, overlap_factor: float) -> Tuple[int, int]:
    L = int(round(Tseg_ms * 1e-3 * fs))
    H = max(1, int(round(L * (1.0 - overlap_factor))))
    return L, H

def sliding_frames(y: np.ndarray, L: int, H: int) -> List[np.ndarray]:
    T = y.shape[-1]
    if T < L: return []
    return [y[..., i:i+L] for i in range(0, T - L + 1, H)]

def stft_window(win_name: str, nfft: int):
    nm = (win_name or "none").lower()
    if nm in ("none", "rect", ""):
        return None
    return getattr(pra.windows, nm)(nfft, flag='asymmetric', length='full')

# -------------------- DoA --------------------

def build_mic_array_from_group_positions(pos_rx_group: np.ndarray,
                                         mic_radius: float):
    mic_pos = pos_rx_group.T  # (3,8)
    mic_center = np.mean(mic_pos[:2, :], axis=1)
    mic_array = pra.beamforming.circular_2D_array(
        center=mic_center, M=8, radius=mic_radius, phi0=np.pi/2
    )
    return mic_array, mic_center

def compute_true_deg(tx_pos_one: np.ndarray, mic_center_xy: np.ndarray) -> float:
    dx = float(tx_pos_one[0] - mic_center_xy[0])
    dy = float(tx_pos_one[1] - mic_center_xy[1])
    return (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

def doa_one_frame(y_frame: np.ndarray, nfft: int, hop: int, win, fs: int,
                  mic_array, algo_name: str) -> Optional[float]:
    try:
        X = np.array([pra.transform.stft.analysis(ch, nfft, hop, win=win) for ch in y_frame])
        X = np.transpose(X, (0, 2, 1))  # (C, F, Tfrm)
        doa = pra.doa.algorithms[algo_name](mic_array, fs=fs, nfft=nfft)
        doa.locate_sources(X)
        return float(np.argmax(doa.grid.values))
    except Exception:
        return None

def doa_sample_from_frames(frames: List[np.ndarray], fs: int,
                           stft_cfg: Dict[str, Any], mic_array, algo_name: str):
    nfft = int(stft_cfg["nfft"]); hop = int(stft_cfg["hop"])
    win = stft_window(stft_cfg["win"], nfft)
    frame_angles: List[float] = []
    for f in frames:
        est = doa_one_frame(f, nfft, hop, win, fs, mic_array, algo_name)
        if est is not None and np.isfinite(est):
            frame_angles.append(est)
    n_frames = len(frames); n_valid = len(frame_angles)
    if n_valid == 0:
        return dict(est_deg=np.nan, var_circ=np.nan, std_circ_deg=np.nan,
                    n_frames=n_frames, n_valid=n_valid, frame_angles_deg=[])
    mu_deg, var_circ, std_circ = circ_stats_deg(frame_angles)
    return dict(est_deg=mu_deg, var_circ=var_circ, std_circ_deg=std_circ,
                n_frames=n_frames, n_valid=n_valid, frame_angles_deg=frame_angles)

# -------------------- サマリ生成（pkl→集計） --------------------

def summarize_from_entries(condition_tag: str,
                           band_name: str, low: float, high: float,
                           Lw: float, Tseg_ms: float, ov: float,
                           win_name: str, nfft_use: int, hop_use: int,
                           entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame([dict(
        seed=e["seed"],
        group=e["group"],
        pred_vs_true_error=e["pred_vs_true_error"],
        pred_vs_gt_error=e["pred_vs_gt_error"],
        var_circ=e["pred"]["var_circ"],
        std_circ_deg=e["pred"]["std_circ_deg"],
    ) for e in entries])

    valid_true = df[np.isfinite(df["pred_vs_true_error"])]
    valid_gt   = df[np.isfinite(df["pred_vs_gt_error"])]

    return dict(
        condition=condition_tag,
        band=band_name, low_hz=low, high_hz=high,
        Lw_sec=Lw, Tseg_ms=float(Tseg_ms), overlap=float(ov),
        stft_win=win_name, stft_nfft=nfft_use, stft_hop=hop_use,
        n_total=len(df),
        n_valid_true=int(len(valid_true)),
        n_valid_gt=int(len(valid_gt)),
        mean_pred_vs_true=float(valid_true["pred_vs_true_error"].mean()) if len(valid_true)>0 else float("nan"),
        std_pred_vs_true=float(valid_true["pred_vs_true_error"].std(ddof=1)) if len(valid_true)>1 else float("nan"),
        mean_pred_vs_gt=float(valid_gt["pred_vs_gt_error"].mean()) if len(valid_gt)>0 else float("nan"),
        std_pred_vs_gt=float(valid_gt["pred_vs_gt_error"].std(ddof=1)) if len(valid_gt)>1 else float("nan"),
        mean_var_circ=float(valid_true["var_circ"].mean()) if len(valid_true)>0 else float("nan"),
        mean_std_circ_deg=float(valid_true["std_circ_deg"].mean()) if len(valid_true)>0 else float("nan"),
    )

def summarize_from_pkl(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    meta = data["meta"]; entries = data["entries"]
    return summarize_from_entries(
        condition_tag=meta["condition"],
        band_name=meta["band"], low=meta["low_hz"], high=meta["high_hz"],
        Lw=meta["Lw_sec"], Tseg_ms=meta["Tseg_ms"], ov=meta["overlap"],
        win_name=meta["stft_win"], nfft_use=meta["stft_nfft"], hop_use=meta["stft_hop"],
        entries=entries
    )

# -------------------- メイン実験 --------------------

def run_grid(cfg: Config, force_cli: bool = False):
    fs = cfg.fs
    force = bool(force_cli or cfg.force)

    H_pred, H_ori, pos_rx, pos_tx = load_npz_freq_ir(cfg.npz)
    groups_pred = groups_of_8(H_pred)
    groups_ori  = groups_of_8(H_ori) if H_ori is not None else [None]*len(groups_pred)
    groups_pos  = groups_of_8(pos_rx) if pos_rx is not None else [None]*len(groups_pred)
    tx_pos      = pos_tx

    root = os.path.expanduser(cfg.outdir); _ensure_dir(root)
    with open(os.path.join(root, "config_effective.yaml"), "w") as f:
        yaml.safe_dump(cfg.__dict__, f, allow_unicode=True, sort_keys=False)

    overall_rows: List[Dict[str, Any]] = []

    for band in cfg.bands:
        low, high = float(band["low"]), float(band["high"])
        band_name = band["name"]
        sos = butter_bandpass_sos(low, high, fs, order=4)

        for Lw in cfg.noise_seconds:
            for Tseg_ms in cfg.segments_ms:
                for ov in cfg.overlap_factors:
                    L, Hhop = seg_hop_samples(fs, Tseg_ms, ov)

                    for st in cfg.stft_grid:
                        nfft_use = int(st["nfft"]); hop_use = int(st["hop"])
                        win_name = str(st["win"]).lower()

                        condition_tag = (
                            f"band_{band_name}"
                            f"/Lw_{Lw:.2f}s"
                            f"/seg_{int(round(Tseg_ms))}ms_ov{int(round(ov*100))}"
                            f"/stft_{win_name}_L{nfft_use}_H{hop_use}"
                        )
                        cond_dir = os.path.join(root, condition_tag)
                        _ensure_dir(cond_dir)
                        pkl_path = os.path.join(cond_dir, "results.pkl")

                        # 既存pklがあり、forceでなければ → 再計算スキップして集計だけ反映
                        if os.path.isfile(pkl_path) and (not force):
                            try:
                                summary = summarize_from_pkl(pkl_path)
                                overall_rows.append(summary)
                                print("[SKIP reuse]", condition_tag, "n_valid_true:", summary["n_valid_true"])
                                continue
                            except Exception as e:
                                print("[WARN] failed to reuse pkl; will recompute:", pkl_path, "err:", repr(e))

                        # ---- ここに生データ全部を溜める（再計算パス or force時） ----
                        all_entries: List[Dict[str, Any]] = []

                        for seed in cfg.seeds:
                            x = white_noise(Lw, fs, seed)

                            for g_idx, (ir_pred, ir_ori) in enumerate(zip(groups_pred, groups_ori), start=1):
                                # mic / true angle（幾何）
                                if groups_pos[g_idx-1] is not None and tx_pos is not None:
                                    pos_rx_g = groups_pos[g_idx-1]
                                    mic_array, mic_center = build_mic_array_from_group_positions(
                                        pos_rx_g, cfg.mic_radius
                                    )
                                    tx_one = tx_pos[(g_idx-1)*8]
                                    true_deg = compute_true_deg(tx_one, mic_center)
                                else:
                                    mic_array = pra.beamforming.circular_2D_array(
                                        center=(0.0,0.0), M=8, radius=cfg.mic_radius, phi0=np.pi/2
                                    )
                                    true_deg = 0.0

                                # pred 観測合成 → BP → フレーミング → DoA集約
                                y_pred = synth_observation(ir_pred, x)
                                y_pred_bp = apply_zero_phase_bp(y_pred, sos)
                                frames_pred = sliding_frames(y_pred_bp, L, Hhop)
                                pred_res = doa_sample_from_frames(
                                    frames_pred, fs,
                                    dict(nfft=nfft_use, hop=hop_use, win=win_name),
                                    mic_array, cfg.algo_name
                                )

                                # ori があれば同じ手順で基準角（gt）
                                gt_est_deg = None
                                if ir_ori is not None:
                                    y_ori = synth_observation(ir_ori, x)
                                    y_ori_bp = apply_zero_phase_bp(y_ori, sos)
                                    frames_ori = sliding_frames(y_ori_bp, L, Hhop)
                                    ori_res = doa_sample_from_frames(
                                        frames_ori, fs,
                                        dict(nfft=nfft_use, hop=hop_use, win=win_name),
                                        mic_array, cfg.algo_name
                                    )
                                    gt_est_deg = ori_res["est_deg"]

                                # 誤差
                                err_true = (angular_error_deg(pred_res["est_deg"], true_deg)
                                            if np.isfinite(pred_res["est_deg"]) else float("nan"))
                                err_gt = (angular_error_deg(pred_res["est_deg"], gt_est_deg)
                                          if (gt_est_deg is not None and np.isfinite(pred_res["est_deg"]) and np.isfinite(gt_est_deg))
                                          else float("nan"))

                                all_entries.append(dict(
                                    seed=seed, group=g_idx,
                                    true_deg=true_deg,
                                    pred=dict(
                                        est_deg=pred_res["est_deg"],
                                        var_circ=pred_res["var_circ"],
                                        std_circ_deg=pred_res["std_circ_deg"],
                                        n_frames=pred_res["n_frames"],
                                        n_valid=pred_res["n_valid"],
                                        frame_angles_deg=pred_res["frame_angles_deg"],
                                    ),
                                    gt=dict(est_deg=gt_est_deg),   # ない場合は None
                                    pred_vs_true_error=err_true,
                                    pred_vs_gt_error=err_gt,
                                ))

                        # ---- 条件ごとの pkl 保存（生データ丸ごと）
                        with open(pkl_path, "wb") as f:
                            pickle.dump(dict(
                                meta=dict(
                                    condition=condition_tag,
                                    band=band_name, low_hz=low, high_hz=high,
                                    Lw_sec=Lw, Tseg_ms=float(Tseg_ms), overlap=float(ov),
                                    stft_win=win_name, stft_nfft=nfft_use, stft_hop=hop_use,
                                    seeds=cfg.seeds
                                ),
                                entries=all_entries
                            ), f)

                        summary = summarize_from_entries(
                            condition_tag, band_name, low, high, Lw, Tseg_ms, ov, win_name, nfft_use, hop_use, all_entries
                        )
                        overall_rows.append(summary)
                        print("[OK]", condition_tag, "n_valid_true:", summary["n_valid_true"])

    # ---- 全条件まとめ（pred_vs_trueの平均 昇順でソート；NaNは末尾）
    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df.sort_values(by=["mean_pred_vs_true"], ascending=True, na_position="last")
    out_csv = os.path.join(os.path.expanduser(cfg.outdir), "summary_all_conditions.csv")
    overall_df.to_csv(out_csv, index=False)
    print("[DONE] summary(sorted):", out_csv)

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="YAML config path")
    ap.add_argument("--force", action="store_true", help="Recompute even if results.pkl exists")
    args = ap.parse_args()
    cfg = load_config(os.path.expanduser(args.cfg))
    run_grid(cfg, force_cli=args.force)

if __name__ == "__main__":
    main()
