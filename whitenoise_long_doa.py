#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STFT-first + Sliding-Window DoA (resume-safe, pkl reuse, --force)

- 100s 白色ノイズ × IR を全長で畳み込み
- 全長STFT -> 時間方向に長さ T_use フレームの窓をスライドして DoA
- 各ウィンドウで 1 つの角度推定（窓中心フレームの index を x として保存）
- IIRバンドパスなし（全帯域）
- pkl があればスキップ、--force で再計算
- 全条件まとめ CSV は mean_pred_vs_true 昇順

deps: numpy, pandas, pyroomacoustics, pyyaml
"""

import os, math, argparse, yaml, pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyroomacoustics as pra


# -------------------- 角度ユーティリティ --------------------

def wrap_deg_signed(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0

def angular_error_deg(a_deg: float, b_deg: float) -> float:
    """|a-b| を [0,180] に短縮した角度差"""
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)

def circ_mean_deg(angles_deg: List[float]) -> Tuple[float, float]:
    """円平均角 [deg], resultant length R [0..1]"""
    if len(angles_deg) == 0:
        return float("nan"), 0.0
    a = np.deg2rad(np.asarray(angles_deg))
    C = float(np.cos(a).sum()); S = float(np.sin(a).sum())
    mu = (np.rad2deg(math.atan2(S, C)) + 360.0) % 360.0
    R = math.hypot(C, S) / max(len(angles_deg), 1)
    return mu, R

def circ_stats_deg(angles_deg: List[float]) -> Tuple[float, float, float]:
    """(円平均角, 円分散=1-R, 円標準偏差[deg])"""
    mu, R = circ_mean_deg(angles_deg)
    var = 1.0 - R
    std_deg = np.rad2deg(np.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12))))) if R > 0 else float("nan")
    return mu, var, std_deg


# -------------------- 入出力 --------------------

@dataclass
class Config:
    npz: str
    fs: int
    seeds: List[int]              # 例: [0]
    long_noise_seconds: float     # 例: 100.0
    stft_grid: List[Dict[str, Any]]   # {nfft, hop, win: "none"/"hann"}
    T_use_list: List[int]         # 例: [16, 32, 64, 128, 256, 512]
    outdir: str
    mic_radius: float = 0.0365
    algo_name: str = "NormMUSIC"
    slide_hop_frames: Optional[int] = None  # 省略時は T_use//4
    force: bool = False

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

def groups_of_8(arr: Optional[np.ndarray]) -> List[Optional[np.ndarray]]:
    if arr is None:
        return []
    N = arr.shape[0]
    if N % 8 != 0:
        raise ValueError(f"N={N} is not divisible by 8.")
    return [arr[g*8:(g+1)*8] for g in range(N // 8)]


# -------------------- 合成・STFT --------------------

def white_noise_long(L_sec: float, fs: int, seed: int) -> np.ndarray:
    Nt = int(round(L_sec * fs))
    rng = np.random.default_rng(seed)
    return rng.standard_normal(Nt).astype(np.float32)

def synth_observation_time(ir_group_freq: np.ndarray, x: np.ndarray) -> np.ndarray:
    """freq-IR (8,F) -> ir_time(8,Nh) として conv 全ch。戻り: y(8,T)"""
    h = np.fft.irfft(ir_group_freq, axis=1).astype(np.float32)  # (8, Nh)
    ys = [np.convolve(x, h_i, mode="full") for h_i in h]
    return np.stack(ys, axis=0).astype(np.float32)              # (8, T)

def stft_window(win_name: str, nfft: int):
    nm = (win_name or "none").lower()
    if nm in ("none", "rect", ""):
        return None
    return getattr(pra.windows, nm)(nfft, flag='asymmetric', length='full')

def stft_full(y: np.ndarray, nfft: int, hop: int, win) -> np.ndarray:
    """y: (C,T) -> X: (C,F,Tfrm)  （全長で一度だけSTFT）"""
    X = np.array([pra.transform.stft.analysis(ch, nfft, hop, win=win) for ch in y])  # (C,Tfrm,F)
    return np.transpose(X, (0, 2, 1)).astype(np.complex64)  # (C,F,Tfrm)


# -------------------- DoA（スライディング窓） --------------------

def build_mic_array_from_group_positions(pos_rx_group: np.ndarray, mic_radius: float):
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

def doa_sliding_over_time(X: np.ndarray, fs: int, nfft: int, mic_array, algo_name: str,
                          T_use: int, hop_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (C,F,Tfrm) の STFT から、長さ T_use の時間窓を hop_frames ごとにスライドして DoA。
    戻り: (angles_deg[Ns], centers_idx[Ns])
    """
    T = X.shape[-1]
    if T < T_use:
        return np.array([]), np.array([])
    doa = pra.doa.algorithms[algo_name](mic_array, fs=fs, nfft=nfft)
    out_angles = []
    centers = []
    for t0 in range(0, T - T_use + 1, hop_frames):
        Xseg = X[:, :, t0:t0 + T_use]
        try:
            doa.locate_sources(Xseg)
            ang = float(np.argmax(doa.grid.values))
            out_angles.append(ang)
            centers.append(t0 + T_use // 2)
        except Exception:
            # スキップ
            continue
    return np.asarray(out_angles, dtype=float), np.asarray(centers, dtype=int)


# -------------------- 実験（グループ単位 → 条件/T_use保存） --------------------

def run_condition_for_group(
    ir_pred: np.ndarray,
    ir_ori: Optional[np.ndarray],
    pos_rx_g: Optional[np.ndarray],
    tx_pos: Optional[np.ndarray],
    x_long: np.ndarray,
    fs: int,
    stft_cfg: Dict[str, Any],
    algo_name: str,
    mic_radius: float,
    T_use: int,
    slide_hop_frames: Optional[int],
) -> Dict[str, Any]:
    # mic & true angle
    if pos_rx_g is not None and tx_pos is not None:
        mic_array, mic_center = build_mic_array_from_group_positions(pos_rx_g, mic_radius)
        true_deg = compute_true_deg(tx_pos[0], mic_center)  # グループ先頭tx
    else:
        mic_array = pra.beamforming.circular_2D_array(center=(0.0, 0.0), M=8, radius=mic_radius, phi0=np.pi/2)
        true_deg = 0.0

    # 観測生成（全長）
    y_pred = synth_observation_time(ir_pred, x_long)   # (8,T)
    y_ori = synth_observation_time(ir_ori, x_long) if ir_ori is not None else None

    # 全長 STFT
    nfft = int(stft_cfg["nfft"]); hop = int(stft_cfg["hop"]); win = stft_window(str(stft_cfg["win"]).lower(), nfft)
    Xp = stft_full(y_pred, nfft, hop, win)            # (C,F,Tfrm)
    Xo = stft_full(y_ori, nfft, hop, win) if y_ori is not None else None

    # スライディングDoA（T_useフレーム窓）
    hop_frames = int(slide_hop_frames) if slide_hop_frames is not None else T_use
    angles_pred, centers_idx = doa_sliding_over_time(Xp, fs, nfft, mic_array, algo_name, T_use, hop_frames)
    if angles_pred.size == 0:
        pred_mu = pred_var = pred_std = float("nan")
        err_true = float("nan")
    else:
        pred_mu, pred_var, pred_std = circ_stats_deg(angles_pred.tolist())
        err_true = angular_error_deg(pred_mu, true_deg)

    gt_mu = float("nan"); err_gt = float("nan")
    if Xo is not None:
        angles_gt, centers_gt = doa_sliding_over_time(Xo, fs, nfft, mic_array, algo_name, T_use, hop_frames)
        if angles_pred.size > 0 and angles_gt.size > 0:
            gt_mu, _, _ = circ_stats_deg(angles_gt.tolist())
            err_gt = angular_error_deg(pred_mu, gt_mu)

    return dict(
        true_deg=true_deg,
        pred=dict(
            angles_deg=angles_pred.tolist(),
            centers=centers_idx.tolist(),
            mu_deg=pred_mu, var_circ=pred_var, std_circ_deg=pred_std,
            n_windows=int(len(centers_idx)), n_valid=int(len(angles_pred)),
        ),
        gt=dict(
            mu_deg=gt_mu,
            angles_deg=angles_gt.tolist(),
            centers=centers_gt.tolist(),        
        ),
        pred_vs_true_error=err_true,
        pred_vs_gt_error=err_gt,
        hop_frames=hop_frames
    )


# -------------------- メインループ --------------------

def run_grid(cfg: Config, force_cli: bool = False):
    fs = cfg.fs
    force = bool(force_cli or cfg.force)

    H_pred, H_ori, pos_rx, pos_tx = load_npz_freq_ir(cfg.npz)
    groups_pred = groups_of_8(H_pred)                     # len = 18
    groups_ori  = groups_of_8(H_ori) if H_ori is not None else [None]*len(groups_pred)
    groups_pos  = groups_of_8(pos_rx) if pos_rx is not None else [None]*len(groups_pred)

    root = os.path.expanduser(cfg.outdir); _ensure_dir(root)
    with open(os.path.join(root, "config_effective.yaml"), "w") as f:
        yaml.safe_dump(cfg.__dict__, f, allow_unicode=True, sort_keys=False)

    overall_rows: List[Dict[str, Any]] = []

    for st in cfg.stft_grid:
        nfft_use = int(st["nfft"]); hop_use = int(st["hop"]); win_name = str(st["win"]).lower()
        stft_tag = f"stft_{win_name}_L{nfft_use}_H{hop_use}"
        stft_dir = os.path.join(root, stft_tag)
        _ensure_dir(stft_dir)

        for seed in cfg.seeds:
            # 長尺ノイズ1本
            x_long = white_noise_long(cfg.long_noise_seconds, fs, seed)

            for T_use in cfg.T_use_list:
                tdir = os.path.join(stft_dir, f"Tuse_{int(T_use)}")
                _ensure_dir(tdir)
                pkl_path = os.path.join(tdir, "results.pkl")

                # 既存pkl → スキップ（再開）
                if os.path.isfile(pkl_path) and (not force):
                    try:
                        with open(pkl_path, "rb") as f:
                            data = pickle.load(f)
                        # summary 行
                        entries = data["entries"]
                        df = pd.DataFrame([
                            dict(
                                seed=e["seed"], group=e["group"],
                                pred_vs_true_error=e["pred_vs_true_error"],
                                pred_vs_gt_error=e["pred_vs_gt_error"],
                                var_circ=e["pred"]["var_circ"],
                                std_circ_deg=e["pred"]["std_circ_deg"],
                            ) for e in entries
                        ])
                        vt = df[np.isfinite(df["pred_vs_true_error"])]
                        vg = df[np.isfinite(df["pred_vs_gt_error"])]
                        overall_rows.append(dict(
                            condition=stft_tag, T_use=int(T_use),
                            n_total=len(df), n_valid_true=int(len(vt)), n_valid_gt=int(len(vg)),
                            mean_pred_vs_true=float(vt["pred_vs_true_error"].mean()) if len(vt)>0 else float("nan"),
                            std_pred_vs_true=float(vt["pred_vs_true_error"].std(ddof=1)) if len(vt)>1 else float("nan"),
                            mean_pred_vs_gt=float(vg["pred_vs_gt_error"].mean()) if len(vg)>0 else float("nan"),
                            std_pred_vs_gt=float(vg["pred_vs_gt_error"].std(ddof=1)) if len(vg)>1 else float("nan"),
                            mean_var_circ=float(vt["var_circ"].mean()) if len(vt)>0 else float("nan"),
                            mean_std_circ_deg=float(vt["std_circ_deg"].mean()) if len(vt)>0 else float("nan"),
                        ))
                        print("[SKIP reuse]", pkl_path)
                        continue
                    except Exception as e:
                        print("[WARN] failed to reuse pkl; will recompute:", pkl_path, "err:", repr(e))

                # ---- 再計算 ----
                all_entries: List[Dict[str, Any]] = []
                for g_idx, (ir_pred, ir_ori, pos_rx_g) in enumerate(zip(groups_pred, groups_ori, groups_pos), start=1):
                    rec = run_condition_for_group(
                        ir_pred=ir_pred, ir_ori=ir_ori,
                        pos_rx_g=pos_rx_g, tx_pos=pos_tx,
                        x_long=x_long, fs=fs,
                        stft_cfg=dict(nfft=nfft_use, hop=hop_use, win=win_name),
                        algo_name=cfg.algo_name, mic_radius=cfg.mic_radius,
                        T_use=int(T_use), slide_hop_frames=cfg.slide_hop_frames,
                    )
                    all_entries.append(dict(
                        seed=seed, group=g_idx,
                        true_deg=rec["true_deg"],
                        pred=rec["pred"],
                        gt=rec["gt"],
                        pred_vs_true_error=rec["pred_vs_true_error"],
                        pred_vs_gt_error=rec["pred_vs_gt_error"],
                        hop_frames=rec["hop_frames"],
                    ))

                with open(pkl_path, "wb") as f:
                    pickle.dump(dict(
                        meta=dict(
                            condition=stft_tag,
                            Tuse=int(T_use),
                            fs=fs,
                            long_noise_seconds=float(cfg.long_noise_seconds),
                            stft_win=win_name, stft_nfft=nfft_use, stft_hop=hop_use,
                            seeds=cfg.seeds,
                            slide_hop_frames=int(cfg.slide_hop_frames) if cfg.slide_hop_frames is not None else None
                        ),
                        entries=all_entries
                    ), f)

                # まとめ行
                df = pd.DataFrame([
                    dict(
                        seed=e["seed"], group=e["group"],
                        pred_vs_true_error=e["pred_vs_true_error"],
                        pred_vs_gt_error=e["pred_vs_gt_error"],
                        var_circ=e["pred"]["var_circ"],
                        std_circ_deg=e["pred"]["std_circ_deg"],
                    ) for e in all_entries
                ])
                vt = df[np.isfinite(df["pred_vs_true_error"])]
                vg = df[np.isfinite(df["pred_vs_gt_error"])]

                overall_rows.append(dict(
                    condition=stft_tag, T_use=int(T_use),
                    n_total=len(df), n_valid_true=int(len(vt)), n_valid_gt=int(len(vg)),
                    mean_pred_vs_true=float(vt["pred_vs_true_error"].mean()) if len(vt)>0 else float("nan"),
                    std_pred_vs_true=float(vt["pred_vs_true_error"].std(ddof=1)) if len(vt)>1 else float("nan"),
                    mean_pred_vs_gt=float(vg["pred_vs_gt_error"].mean()) if len(vg)>0 else float("nan"),
                    std_pred_vs_gt=float(vg["pred_vs_gt_error"].std(ddof=1)) if len(vg)>1 else float("nan"),
                    mean_var_circ=float(vt["var_circ"].mean()) if len(vt)>0 else float("nan"),
                    mean_std_circ_deg=float(vt["std_circ_deg"].mean()) if len(vt)>0 else float("nan"),
                ))
                print("[OK]", stft_tag, f"T_use={T_use}", "n_valid_true:", len(vt))

    # summary 保存（pred_vs_true 平均の昇順）
    overall_df = pd.DataFrame(overall_rows)
    overall_df = overall_df.sort_values(by=["mean_pred_vs_true"], ascending=True, na_position="last")
    out_csv = os.path.join(root, "summary_all_conditions.csv")
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
