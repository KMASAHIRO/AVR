import os
import re
import glob
import pickle
import argparse
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator


# ====== 参照ロジック準拠のイベント探索 ======
def find_tensorboard_event_file(tensorboard_logdir: str, relative_subpath: str, expname: str) -> str:
    """
    tensorboard_logdir/relative_subpath/expname/**/events.out.tfevents.*
    という構成からイベントファイルを検索（最新更新を優先して選択）
    """
    exp_dir = os.path.join(tensorboard_logdir, relative_subpath, expname)
    pattern = os.path.join(exp_dir, '**', 'events.out.tfevents.*')
    event_files = glob.glob(pattern, recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {exp_dir}")
    event_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return event_files[0]


# ====== 内部ユーティリティ ======
def _load_scalar_sum_by_step(ea: event_accumulator.EventAccumulator, prefix: str) -> Dict[int, float]:
    """
    EventAccumulatorから prefix で始まる scalar を全て足し合わせ、{step: sum} を返す。
    例: prefix="train_loss/" or "test_loss/"
    """
    tags = [t for t in ea.Tags().get("scalars", []) if t.startswith(prefix)]
    acc = defaultdict(float)
    for tag in tags:
        for ev in ea.Scalars(tag):
            acc[int(ev.step)] += float(ev.value)
    return dict(acc)


def _closest_value_at_step(step_to_val: Dict[int, float], target_step: int) -> Optional[float]:
    """
    step が完全一致しない場合に備え、最も近い step の値を返す（無ければ None）。
    """
    if not step_to_val:
        return None
    if target_step in step_to_val:
        return step_to_val[target_step]
    keys = np.array(list(step_to_val.keys()), dtype=int)
    idx = int(np.argmin(np.abs(keys - target_step)))
    return step_to_val[int(keys[idx])]


def _load_mean_error_and_iter_from_pkl(
    pkl_path: str, algo_name: str, error_key: str
) -> Optional[Tuple[float, int]]:
    """
    val_iterN.pkl を読み、指定アルゴリズムの誤差リスト（None除去）の **平均値** と、
    その pkl の iter 番号（ファイル名から取得）を返す。
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        errs = data.get(algo_name, {}).get(error_key, [])
        errs = [e for e in errs if e is not None]
        if len(errs) == 0:
            return None
        mean_err = float(np.mean(errs))  # ★ 平均で集計
        m = re.search(r"val_iter(\d+)\.pkl$", os.path.basename(pkl_path))
        it = int(m.group(1)) if m else -1
        return (mean_err, it)
    except Exception:
        return None


# ====== 本体：横軸=trial、DoA平均誤差と、その時点のtrain/val lossを描画 & CSV出力 ======
def plot_trialwise_doa_mean_and_losses_discovered(
    tensorboard_logdir: str,
    logdir: str,
    trial_names: List[str],
    csv_output_path: str,
    figure_output_path: str = "trialwise_doa_mean_and_loss.png",
    algo_name: str = "NormMUSIC",
    error_key: str = "pred_vs_gt_error",
    ylim_doa: Tuple[float, float] = (0.0, 120.0),
    skip_missing: bool = False,
) -> None:
    """
    - 横軸: trial（例: Real_exp_param_{i}_1）
    - 各 trial について:
        - DoA結果: {logdir}/{trial}/doa_results/val_iter*.pkl を走査
          → **各pklの平均誤差**を計算 → その平均が最小の iter を採用
        - TBイベント: find_tensorboard_event_file() を用いて探索
        - 採用 iter に最も近い step の train/val loss を取得
    - グラフ保存 + CSV保存
    - skip_missing=True のとき、必須情報が欠ける trial は **完全に除外**（CSV/図に出さない）
    """
    os.makedirs(os.path.dirname(csv_output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(figure_output_path) or ".", exist_ok=True)

    # "logs/real_exp" -> "real_exp"（先頭セグメントを除去）
    logdir_norm = logdir.replace("\\", "/").rstrip("/")
    parts = logdir_norm.split("/")
    if len(parts) >= 2:
        logdir_subpath = "/".join(parts[1:])
    else:
        logdir_subpath = logdir_norm

    rows = []
    trial_idx = []
    trial_labels = []
    doa_means = []
    train_losses_at_mean = []
    val_losses_at_mean = []

    next_index = 1  # スキップ時も連番を綺麗に保つための index

    for expname in trial_names:
        base_path = os.path.join(logdir, expname)
        doa_save_dir = os.path.join(base_path, "doa_results")

        # --- DoA 平均誤差 & iter 探索（min-of-mean） ---
        pkl_files = sorted(glob.glob(os.path.join(doa_save_dir, "val_iter*.pkl")))
        best_err, best_iter = None, None
        for p in pkl_files:
            res = _load_mean_error_and_iter_from_pkl(p, algo_name, error_key)
            if res is None:
                continue
            err, it = res
            if best_err is None or err < best_err:
                best_err, best_iter = err, it

        # --- TensorBoard イベント探索（参照ロジック） ---
        event_path = None
        try:
            event_path = find_tensorboard_event_file(tensorboard_logdir, logdir_subpath, expname)
        except FileNotFoundError:
            event_path = None

        # --- 採用 iter の train/val loss を取得 ---
        tr_loss_at_mean, va_loss_at_mean = None, None
        if best_iter is not None and event_path is not None:
            try:
                ea = event_accumulator.EventAccumulator(event_path)
                ea.Reload()
                train_sum = _load_scalar_sum_by_step(ea, "train_loss/")
                val_sum = _load_scalar_sum_by_step(ea, "test_loss/")
                tr_loss_at_mean = _closest_value_at_step(train_sum, best_iter)
                va_loss_at_mean = _closest_value_at_step(val_sum, best_iter)
            except Exception:
                pass

        # --- スキップ判定 ---
        missing = (
            (best_err is None) or
            (best_iter is None) or
            (event_path is None) or
            (tr_loss_at_mean is None and va_loss_at_mean is None)
        )
        if skip_missing and missing:
            continue  # trial 丸ごと除外

        # --- 配列・CSV行へ格納（欠損は NaN） ---
        trial_idx.append(next_index)
        trial_labels.append(expname)
        doa_means.append(best_err if best_err is not None else np.nan)
        train_losses_at_mean.append(tr_loss_at_mean if tr_loss_at_mean is not None else np.nan)
        val_losses_at_mean.append(va_loss_at_mean if va_loss_at_mean is not None else np.nan)

        rows.append({
            "trial_index": next_index,
            "trial_name": expname,
            "val_iter_at_doa_mean": best_iter if best_iter is not None else np.nan,
            "doa_mean_error_deg": best_err if best_err is not None else np.nan,
            "train_loss_at_doa_mean": tr_loss_at_mean if tr_loss_at_mean is not None else np.nan,
            "val_loss_at_doa_mean": va_loss_at_mean if va_loss_at_mean is not None else np.nan,
            "doa_dir": doa_save_dir,
            "event_file": event_path or "",
        })
        next_index += 1

    if len(rows) == 0:
        print("[WARN] No trials remained (all missing or empty). CSV/plot not generated.")
        return

    # --- CSV出力 ---
    df = pd.DataFrame(rows)
    df.to_csv(csv_output_path, index=False)

    # --- 描画（横軸=trial index） ---
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左軸：Loss
    ax1.set_xlabel("Trial Index")
    ax1.set_ylabel("Loss", color="black")
    ax1.plot(trial_idx, train_losses_at_mean, marker="o", label="Train Loss @ DoA mean")
    ax1.plot(trial_idx, val_losses_at_mean, marker="o", label="Val Loss @ DoA mean")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(False)  # 左軸グリッドはOFF

    # 右軸：DoA（緑固定）
    ax2 = ax1.twinx()
    ax2.set_ylabel("DoA Mean Error (°)")
    ax2.plot(trial_idx, doa_means, marker="s", label="DoA mean", linewidth=2, color="green")
    ax2.set_ylim(*ylim_doa)
    ax2.tick_params(axis="y")
    ax2.grid(True, axis="y", alpha=0.3)  # 右軸の目盛りに揃うグリッド

    # 凡例（左右統合）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # x 軸は trial index（1,2,3,...）
    plt.xticks(trial_idx)

    plt.title("Trial-wise DoA Mean Error and Loss (at DoA mean)")
    plt.tight_layout()
    plt.savefig(figure_output_path, dpi=300)
    plt.close()

# ====== CLI ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot trial-wise DoA mean error and losses (trial axis), with optional skipping."
    )
    parser.add_argument("--tensorboard_logdir", type=str, default="tensorboard_logs",
                        help="Base directory of TensorBoard logs (e.g., tensorboard_logs)")
    parser.add_argument("--logdir", type=str, default="logs/real_exp",
                        help="Base logdir that contains each trial dir (e.g., logs/real_exp)")
    parser.add_argument("--trial_begin", type=int, default=1)
    parser.add_argument("--trial_end", type=int, default=64)
    parser.add_argument("--trial_fmt", type=str, default="Real_exp_param_{i}_1",
                        help='Python format string for trial name (use "{i}")')
    parser.add_argument("--csv_output", type=str,
                        default="avr_tuning_logs/real_exp/trialwise_doa_min_and_loss.csv")
    parser.add_argument("--fig_output", type=str,
                        default="avr_tuning_logs/real_exp/trialwise_doa_min_and_loss.png")
    parser.add_argument("--algo_name", type=str, default="NormMUSIC")
    parser.add_argument("--error_key", type=str, default="pred_vs_gt_error")
    parser.add_argument("--doa_ylim", type=float, nargs=2, default=(0.0, 120.0))
    parser.add_argument("--skip_missing", action="store_true",
                        help="Skip trials without DoA/event/losses instead of filling NaN.")

    args = parser.parse_args()

    # trial 名列の生成
    trial_names = [args.trial_fmt.format(i=i) for i in range(args.trial_begin, args.trial_end + 1)]

    # 出力先の作成
    os.makedirs(os.path.dirname(args.csv_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.fig_output) or ".", exist_ok=True)

    plot_trialwise_doa_mean_and_losses_discovered(
        tensorboard_logdir=args.tensorboard_logdir,
        logdir=args.logdir,
        trial_names=trial_names,
        csv_output_path=args.csv_output,
        figure_output_path=args.fig_output,
        algo_name=args.algo_name,
        error_key=args.error_key,
        ylim_doa=tuple(args.doa_ylim),
        skip_missing=args.skip_missing,
    )
