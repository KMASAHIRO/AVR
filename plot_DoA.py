import os
import re
import matplotlib.pyplot as plt
import pickle
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

def plot_doa_error_over_iterations(
    val_result_dir: str,
    save_dir: str,
    algo_name: str = "NormMUSIC",
    error_key: str = "pred_vs_gt_error",
    output_path: str = "doa_error_plot.png",
    run_doa_func=None
):
    """
    val_iter*.npz を順次処理し、DoA誤差を描画・保存。

    Parameters:
        val_result_dir (str): val_iter*.npz を含む入力ディレクトリ
        save_dir (str): 処理結果（.pkl）の保存先ディレクトリ
        algo_name (str): 処理対象アルゴリズム名（例: 'NormMUSIC'）
        error_key (str): プロットする誤差キー（例: 'pred_vs_gt_error'）
        output_path (str): PNG保存先のパス
        run_doa_func (callable): run_doa_on_npz 互換の処理関数
    """
    assert run_doa_func is not None, "run_doa_func must be provided"
    os.makedirs(save_dir, exist_ok=True)

    npz_files = sorted([
        f for f in os.listdir(val_result_dir)
        if re.match(r"val_iter\d+\.npz", f)
    ], key=lambda x: int(re.findall(r"\d+", x)[0]))

    iters = []
    mean_errors = []

    for npz_file in npz_files:
        iter_num = int(re.findall(r"\d+", npz_file)[0])
        npz_path = os.path.join(val_result_dir, npz_file)
        pkl_filename = os.path.splitext(npz_file)[0] + ".pkl"
        pkl_path = os.path.join(save_dir, pkl_filename)

        if not os.path.exists(pkl_path):
            run_doa_func(
                npz_path=npz_path,
                fs=16000,
                n_fft=512,
                mic_radius=0.0365,
                algo_names=[algo_name],
                save_path=pkl_path
            )

        # 読み込み・平均誤差算出
        with open(pkl_path, "rb") as f:
            doa_results = pickle.load(f)

        errors = [
            e for e in doa_results[algo_name][error_key]
            if e is not None
        ]
        if errors:
            iters.append(iter_num)
            mean_errors.append(sum(errors) / len(errors))

    # === ステップをエポックに変換 ===
    first_iter = min(iters)
    epochs = [iter / first_iter for iter in iters]

    # グラフ描画
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mean_errors, label=f'{algo_name}: {error_key}')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Angular Error (°)")
    plt.title(f"{algo_name} {error_key.replace('_', ' ').title()} Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    plot_doa_error_over_iterations(
        val_result_dir="/home/ach17616qc/logs/real_exp/Real_exp_param_1_1/val_result",
        save_dir="/home/ach17616qc/logs/real_exp/Real_exp_param_1_1/doa_results",
        algo_name="NormMUSIC",
        error_key="pred_vs_gt_error",
        output_path="/home/ach17616qc/logs/real_exp/Real_exp_param_1_1/normmusic_pred_vs_gt_error.png",
        run_doa_func=run_doa_on_npz
    )

