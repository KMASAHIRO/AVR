# doa_compare_stft_conditions.py
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

# ---------- DoA 実行に必要な依存 ----------
import math
import pyroomacoustics as pra
import torch

# ========== 基本ユーティリティ ==========
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _parse_condition_tag(cond: str) -> Tuple[Optional[str], int, int]:
    """
    'doa_<win>_L<nfft>_H<hop>' -> (win_name, n_fft, hop)
    win='rect' は None 扱い
    """
    m = re.match(r"^doa_([^_]+)_L(\d+)_H(\d+)$", cond)
    if not m:
        raise ValueError(f"Invalid condition tag: {cond}")
    win_raw, nfft, hop = m.group(1), int(m.group(2)), int(m.group(3))
    win = None if win_raw.lower() in ["rect", "none", ""] else win_raw.lower()
    return win, nfft, hop

def _iter_from_fname(path: str) -> int:
    m = re.search(r"val_iter(\d+)\.", os.path.basename(path))
    return int(m.group(1)) if m else -1

# ========== 角度誤差 ==========
def angular_error_deg(a_deg, b_deg):
    return abs((a_deg - b_deg + 180) % 360 - 180)

# ========== PRA STFT/窓 ==========
def _make_window(win_name: Optional[str], n_fft: int):
    if win_name is None:
        return None
    return getattr(pra.windows, win_name)(n_fft, flag='asymmetric', length='full')

def _compute_stft(signals: np.ndarray, n_fft: int, hop: int, win):
    # signals: (C, T)
    return np.array([
        pra.transform.stft.analysis(sig, n_fft, hop, win=win)
        for sig in signals
    ])  # -> (C, n_frames, n_freq)

# ========== DoA 実行（1つの npz に対して） ==========
def run_doa_on_npz_eval(
    npz_path: str,
    fs: int,
    n_fft: int,
    hop: int,
    win_name: Optional[str],
    algo_name: str = "NormMUSIC",
    mic_radius: float = 0.0365,
) -> Dict[str, List[Optional[float]]]:
    """
    既存の npz（pred_sig, ori_sig 等）を読み、指定 STFT で DoA を実行。
    戻り値: {"pred_vs_true_error": [...]}（グループごとに1要素）
    """
    data = np.load(npz_path)
    pred_sig = data['pred_sig']        # (N, T) complex64 (freq-domain IR)
    ori_sig  = data['ori_sig']         # (N, T)
    position_rx = data['position_rx']  # (N, 3)
    position_tx = data['position_tx']  # (N, 3)

    N = pred_sig.shape[0]
    M = 8
    G = N // M

    win = _make_window(win_name, n_fft)
    results = { "pred_vs_true_error": [] }

    for g in range(G):
        idxs = np.arange(g*M, (g+1)*M)
        pred_group = pred_sig[idxs]
        ori_group  = ori_sig[idxs]
        rx_pos = position_rx[idxs]
        tx_pos = position_tx[idxs][0]

        mic_pos = rx_pos.T
        mic_center = np.mean(mic_pos[:2, :], axis=1)

        mic_array = pra.beamforming.circular_2D_array(
            center=mic_center, M=M, radius=mic_radius, phi0=np.pi/2
        )

        dx, dy = tx_pos[0] - mic_center[0], tx_pos[1] - mic_center[1]
        true_deg = (np.degrees(math.atan2(dy, dx)) % 360)

        # freq-IR -> time-IR
        pred_time = torch.real(torch.fft.irfft(torch.tensor(pred_group), dim=-1)).cpu().numpy().astype(np.float32)
        ori_time  = torch.real(torch.fft.irfft(torch.tensor(ori_group),  dim=-1)).cpu().numpy().astype(np.float32)

        # STFT
        X_pred = _compute_stft(pred_time, n_fft, hop, win)    # (C, Tfrm, F)
        X_ori  = _compute_stft(ori_time,  n_fft, hop, win)
        X_pred = np.transpose(X_pred, (0, 2, 1))              # (C, F, Tfrm)
        X_ori  = np.transpose(X_ori,  (0, 2, 1))

        try:
            doa_pred = pra.doa.algorithms[algo_name](mic_array, fs=fs, nfft=n_fft)
            doa_pred.locate_sources(X_pred)

            doa_gt = pra.doa.algorithms[algo_name](mic_array, fs=fs, nfft=n_fft)
            doa_gt.locate_sources(X_ori)

            if algo_name == 'FRIDA':
                pred_deg = np.argmax(np.abs(doa_pred._gen_dirty_img()))
            else:
                pred_deg = np.argmax(doa_pred.grid.values)

            err_pred_vs_true = angular_error_deg(pred_deg, true_deg)
            results["pred_vs_true_error"].append(err_pred_vs_true)

        except Exception:
            results["pred_vs_true_error"].append(None)

    return results

# ========== 1 条件×1 trial：npz -> pkl を作る ==========
def compute_condition_for_trial(
    tdir: str,
    cond_tag: str,
    algo_name: str,
    error_key: str,
    fs: int,
    npz_dirname: str,
    npz_glob: str,
    force: bool,
) -> Tuple[List[int], List[float]]:
    """
    tdir/npz_dirname/npz_glob を走査し、各 npz に対して DoA を実行し
    tdir/doa_compare_stft_conditions/<cond_tag>/val_iter*.pkl を作成。
    そのうえで (iters[], mean_errors[]) を返す。
    """
    win_name, n_fft, hop = _parse_condition_tag(cond_tag)
    cond_dir = os.path.join(tdir, "doa_compare_stft_conditions", cond_tag)
    _ensure_dir(cond_dir)

    # 入力 npz を探索
    in_root1 = os.path.join(tdir, npz_dirname)
    in_root2 = tdir
    if os.path.isdir(in_root1):
        npz_paths = sorted(glob.glob(os.path.join(in_root1, npz_glob)))
    else:
        npz_paths = sorted(glob.glob(os.path.join(in_root2, npz_glob)))

    xs, ys = [], []

    for npz_path in npz_paths:
        it = _iter_from_fname(npz_path)
        out_pkl = os.path.join(cond_dir, f"val_iter{it}.pkl")
        if (not force) and os.path.isfile(out_pkl):
            # 既存 pkl から読み込み
            try:
                with open(out_pkl, "rb") as f:
                    data = pickle.load(f)
                errs = data.get(algo_name, {}).get(error_key, [])
                errs = [e for e in errs if e is not None]
                if errs:
                    xs.append(it)
                    ys.append(float(np.mean(errs)))
                continue
            except Exception:
                pass  # 壊れていたら再計算へ続行

        # DoA 計算
        res = run_doa_on_npz_eval(
            npz_path=npz_path,
            fs=fs,
            n_fft=n_fft,
            hop=hop,
            win_name=win_name,
            algo_name=algo_name,
        )

        # 保存形式：{algo_name: {error_key: list}}
        out_data = {algo_name: {error_key: res["pred_vs_true_error"]}}
        with open(out_pkl, "wb") as f:
            pickle.dump(out_data, f)

        errs = [e for e in res["pred_vs_true_error"] if e is not None]
        if errs:
            xs.append(it)
            ys.append(float(np.mean(errs)))

    # iter 昇順に整列
    if xs:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
    return xs, ys

# ========== メイン（trialごとに DoA 計算 → 図保存；最後に条件ごとのまとめ図 & CSV） ==========
def run_trialwise(
    base_logdir: str,
    trial_names: List[str],
    conditions: List[str],      # ["doa_rect_L128_H32", ...]
    algo_name: str,
    error_key: str,
    ylim_doa: Tuple[float, float],
    combined_minplot_path: str,
    csv_output_path: str,
    fs: int,
    npz_dirname: str,
    npz_glob: str,
    force: bool,
):
    cond_to_trial_idx: Dict[str, List[int]] = {c: [] for c in conditions}
    cond_to_min_errors: Dict[str, List[float]] = {c: [] for c in conditions}
    csv_rows = []

    for t_index, tname in enumerate(trial_names, start=1):
        tdir = os.path.join(base_logdir, tname)
        base_cond_dir = os.path.join(tdir, "doa_compare_stft_conditions")
        _ensure_dir(base_cond_dir)

        # 条件ごとの epoch 曲線を 1 枚に
        K = len(conditions)
        ncols = min(3, K)
        nrows = (K + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols+1.5, 3.8*nrows), squeeze=False)

        for k, cond_tag in enumerate(conditions):
            ax = axes[k // ncols][k % ncols]

            xs, ys = compute_condition_for_trial(
                tdir=tdir,
                cond_tag=cond_tag,
                algo_name=algo_name,
                error_key=error_key,
                fs=fs,
                npz_dirname=npz_dirname,
                npz_glob=npz_glob,
                force=force,
            )

            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5)
                local_min = float(np.min(ys))
                title = f"{cond_tag} | min={local_min:.2f}°"
            else:
                local_min = np.nan
                title = f"{cond_tag} | min=NaN"

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("epoch(iter)")
            ax.set_ylabel("mean DoA error (deg)")
            ax.set_ylim(*ylim_doa)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="y", alpha=0.3)

            # trial→条件の min 蓄積（まとめ図/CSV 用）
            cond_to_trial_idx[cond_tag].append(t_index)
            cond_to_min_errors[cond_tag].append(local_min)
            csv_rows.append({
                "trial_index": t_index,
                "trial_name": tname,
                "condition": cond_tag,
                "min_mean_error_deg": local_min,
            })

        # 余白 subplot 削除
        for i in range(K, nrows*ncols):
            fig.delaxes(axes[i // ncols][i % ncols])

        plt.suptitle(f"Trial: {tname}  (epoch curves by STFT conditions)", y=1.02)
        plt.tight_layout()
        out_png = os.path.join(base_cond_dir, "epoch_curves_grid.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    # 条件ごとの trial 軸まとめ図
    _ensure_dir(os.path.dirname(combined_minplot_path) or ".")
    K = len(conditions)
    ncols = min(3, K)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols+1.5, 3.8*nrows), squeeze=False)

    for k, cond_tag in enumerate(conditions):
        ax = axes[k // ncols][k % ncols]
        x = cond_to_trial_idx[cond_tag]
        y = cond_to_min_errors[cond_tag]
        ax.plot(x, y, marker="s", linewidth=1.5, label="min mean DoA err")
        ax.set_xlabel("Trial Index")
        ax.set_ylabel("min mean DoA error (deg)")
        ax.set_ylim(*ylim_doa)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        gmin = float(np.nanmin(y)) if np.any(~np.isnan(y)) else np.nan
        ax.set_title(f"{cond_tag} | min={gmin:.2f}°" if not np.isnan(gmin) else f"{cond_tag} | min=NaN")
        ax.legend(loc="upper right", fontsize=8)

    for i in range(K, nrows*ncols):
        fig.delaxes(axes[i // ncols][i % ncols])

    plt.suptitle("DoA min(mean) across trials per STFT condition", y=1.02)
    plt.tight_layout()
    plt.savefig(combined_minplot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # CSV 保存
    _ensure_dir(os.path.dirname(csv_output_path) or ".")
    pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False)

# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial-wise STFT-condition DoA pipeline (compute & plot, no losses).")
    parser.add_argument("--logdir", type=str, default="logs/real_exp",
                        help="Base dir that contains each trial dir")
    parser.add_argument("--trial_begin", type=int, default=1)
    parser.add_argument("--trial_end", type=int, default=112)
    parser.add_argument("--trial_fmt", type=str, default="Real_exp_param_{i}_1")

    parser.add_argument("--conditions", type=str, nargs="+", required=True,
                        help='Condition subdirs under {tdir}/doa_compare_stft_conditions/, '
                             'format: "doa_<win>_L<nfft>_H<hop>", e.g., "doa_rect_L128_H32" "doa_hann_L256_H64"')
    parser.add_argument("--algo_name", type=str, default="NormMUSIC")
    parser.add_argument("--error_key", type=str, default="pred_vs_true_error")
    parser.add_argument("--ylim_doa", type=float, nargs=2, default=(0.0, 120.0))
    parser.add_argument("--combined_minplot_path", type=str,
                        default="avr_tuning_logs/real_exp/doa_compare_stft_conditions.png")
    parser.add_argument("--csv_output", type=str,
                        default="avr_tuning_logs/real_exp/doa_compare_stft_conditions.csv")

    # 入力 npz の場所/パターン（必要に応じて調整）
    parser.add_argument("--npz_dirname", type=str, default="val_result",
                        help="Where val_iter*.npz live under each tdir; fallback to tdir if not found")
    parser.add_argument("--npz_glob", type=str, default="val_iter*.npz")
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite existing pkl")

    args = parser.parse_args()

    trials = [args.trial_fmt.format(i=i) for i in range(args.trial_begin, args.trial_end + 1)]
    run_trialwise(
        base_logdir=args.logdir,
        trial_names=trials,
        conditions=args.conditions,
        algo_name=args.algo_name,
        error_key=args.error_key,
        ylim_doa=tuple(args.ylim_doa),
        combined_minplot_path=args.combined_minplot_path,
        csv_output_path=args.csv_output,
        fs=args.fs,
        npz_dirname=args.npz_dirname,
        npz_glob=args.npz_glob,
        force=args.force,
    )
