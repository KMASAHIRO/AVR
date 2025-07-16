import os
import re
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import numpy as np
import torch
import math
import pyroomacoustics as pra

def angular_error_deg(est_deg, ref_deg):
    return min(abs(est_deg - ref_deg), 360 - abs(est_deg - ref_deg))

def run_doa_on_npz(
    npz_path,
    fs=16000,
    n_fft=512,
    mic_radius=0.0365,
    algo_names=None,
    save_path=None
):
    if algo_names is None:
        algo_names = ['MUSIC', 'NormMUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA']

    data = np.load(npz_path)

    pred_sig = data['pred_sig']        # (N, T), complex64
    ori_sig = data['ori_sig']          # (N, T), complex64
    position_rx = data['position_rx']  # (N, 3)
    position_tx = data['position_tx']  # (N, 3)

    N = pred_sig.shape[0]
    M = 8  # マイク数
    G = N // M  # グループ数

    doa_results = {algo: {
        "true_deg": [],
        "pred_deg": [],
        "gt_deg": [],
        "pred_vs_gt_error": [],
        "pred_vs_true_error": [],
        "gt_vs_true_error": []
    } for algo in algo_names}

    for g in range(G):
        idxs = np.arange(g * M, (g + 1) * M)
        pred_group = pred_sig[idxs]
        ori_group = ori_sig[idxs]
        rx_pos = position_rx[idxs]
        tx_pos = position_tx[idxs][0]

        mic_pos = rx_pos.T
        mic_center = np.mean(mic_pos[:2, :], axis=1)

        mic_array = pra.beamforming.circular_2D_array(
            center=mic_center,
            M=M,
            radius=mic_radius,
            phi0=np.pi / 2
        )

        dx, dy = tx_pos[0] - mic_center[0], tx_pos[1] - mic_center[1]
        true_rad = math.atan2(dy, dx)
        true_deg = np.degrees(true_rad) % 360

        pred_time = torch.real(torch.fft.irfft(torch.tensor(pred_group), dim=-1)).cpu().numpy()
        ori_time = torch.real(torch.fft.irfft(torch.tensor(ori_group), dim=-1)).cpu().numpy()

        def compute_stft(signals):
            return np.array([
                pra.transform.stft.analysis(sig, n_fft, n_fft // 2)
                for sig in signals
            ])

        X_pred = compute_stft(pred_time)
        X_ori = compute_stft(ori_time)

        X_pred = np.transpose(X_pred, (0, 2, 1))
        X_ori = np.transpose(X_ori, (0, 2, 1))

        for algo in algo_names:
            try:
                doa_pred = pra.doa.algorithms[algo](mic_array, fs=fs, nfft=n_fft)
                doa_pred.locate_sources(X_pred)

                doa_ori = pra.doa.algorithms[algo](mic_array, fs=fs, nfft=n_fft)
                doa_ori.locate_sources(X_ori)

                if algo == 'FRIDA':
                    pred_deg = np.argmax(np.abs(doa_pred._gen_dirty_img()))
                    gt_deg = np.argmax(np.abs(doa_ori._gen_dirty_img()))
                else:
                    pred_deg = np.argmax(doa_pred.grid.values)
                    gt_deg = np.argmax(doa_ori.grid.values)

                err_pred_vs_gt = angular_error_deg(pred_deg, gt_deg)
                err_pred_vs_true = angular_error_deg(pred_deg, true_deg)
                err_gt_vs_true = angular_error_deg(gt_deg, true_deg)

                doa_results[algo]["true_deg"].append(true_deg)
                doa_results[algo]["pred_deg"].append(pred_deg)
                doa_results[algo]["gt_deg"].append(gt_deg)
                doa_results[algo]["pred_vs_gt_error"].append(err_pred_vs_gt)
                doa_results[algo]["pred_vs_true_error"].append(err_pred_vs_true)
                doa_results[algo]["gt_vs_true_error"].append(err_gt_vs_true)

            except Exception as e:
                for key in doa_results[algo]:
                    doa_results[algo][key].append(None)

    # 結果保存
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(doa_results, f)

def plot_loss_and_doa_over_epochs(
    log_path: str,
    doa_npz_dir: str,
    doa_save_dir: str,
    algo_name: str = "NormMUSIC",
    error_key: str = "pred_vs_gt_error",
    output_path: str = "loss_and_doa_plot.png",
    run_doa_func=None,
    fs: int = 16000,
    n_fft: int = 512,
    mic_radius: float = 0.0365,
):
    """
    LossログとDoA誤差を同時に読み込み、1つのグラフに描画・保存する。

    Parameters:
        log_path (str): TensorBoard eventファイルのパス
        doa_npz_dir (str): val_iter*.npz があるディレクトリ
        doa_save_dir (str): .pkl 結果保存先
        algo_name (str): DoAアルゴリズム名（例: "NormMUSIC"）
        error_key (str): DoA誤差のキー（例: "pred_vs_gt_error"）
        output_path (str): 出力グラフファイル名
        run_doa_func (callable): run_doa_on_npz に準拠した関数
        fs, n_fft, mic_radius: DoA推定のパラメータ
    """
    assert run_doa_func is not None, "run_doa_func must be provided"
    os.makedirs(doa_save_dir, exist_ok=True)

    # === Lossの読み取り ===
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    train_loss_tags = [tag for tag in ea.Tags()['scalars'] if tag.startswith('train_loss/') and tag != 'train_loss']
    test_loss_tags = [tag for tag in ea.Tags()['scalars'] if tag.startswith('test_loss/')]

    def accumulate_tags(tags):
        acc = defaultdict(float)
        for tag in tags:
            for event in ea.Scalars(tag):
                acc[event.step] += event.value
        return acc

    train_loss_sum = accumulate_tags(train_loss_tags)
    test_loss_sum = accumulate_tags(test_loss_tags)

    train_steps, train_values = zip(*sorted(train_loss_sum.items()))
    test_steps, test_values = zip(*sorted(test_loss_sum.items()))

    first_step = min(train_steps)
    train_epochs = [s / first_step for s in train_steps]
    test_epochs = [s / first_step for s in test_steps]

    # === DoA処理 ===
    npz_files = sorted([
        f for f in os.listdir(doa_npz_dir)
        if re.match(r"val_iter\d+\.npz", f)
    ], key=lambda x: int(re.findall(r"\d+", x)[0]))

    doa_epochs = []
    doa_errors = []

    for npz_file in npz_files:
        iter_num = int(re.findall(r"\d+", npz_file)[0])
        epoch = iter_num / first_step

        npz_path = os.path.join(doa_npz_dir, npz_file)
        pkl_path = os.path.join(doa_save_dir, npz_file.replace(".npz", ".pkl"))

        if not os.path.exists(pkl_path):
            run_doa_func(
                npz_path=npz_path,
                fs=fs,
                n_fft=n_fft,
                mic_radius=mic_radius,
                algo_names=[algo_name],
                save_path=pkl_path
            )

        with open(pkl_path, "rb") as f:
            doa_results = pickle.load(f)

        errors = [e for e in doa_results[algo_name][error_key] if e is not None]
        if errors:
            mean_error = sum(errors) / len(errors)
            doa_epochs.append(epoch)
            doa_errors.append(mean_error)

    # === 描画 ===
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss（左軸）
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.plot(train_epochs, train_values, label="Train Loss", color="blue")
    ax1.plot(test_epochs, test_values, label="Test Loss", color="orange")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(True)

    # DoA（右軸）
    ax2 = ax1.twinx()
    ax2.set_ylabel("DoA Error (°)")
    ax2.plot(doa_epochs, doa_errors, label="DoA Error", color="green")
    ax2.set_ylim(0, 120)
    ax2.tick_params(axis='y')

    # 凡例統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"Loss and DoA Error ({error_key}) over Epochs")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    plot_loss_and_doa_over_epochs(
        log_path="/home/ach17616qc/tensorboard_logs/pra/Pra_param_1_1/0707-153432/events.out.tfevents.1751870073.hnode002.198220.0",
        doa_npz_dir="/home/ach17616qc/logs/pra/Pra_param_1_1/val_result",
        doa_save_dir="/home/ach17616qc/logs/pra/Pra_param_1_1/doa_results",
        algo_name="NormMUSIC",
        error_key="pred_vs_gt_error",
        output_path="/home/ach17616qc/logs/pra/Pra_param_1_1/loss_and_doa_plot.png",
        run_doa_func=run_doa_on_npz
    )
