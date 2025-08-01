import numpy as np
import torch
import math
import os
import pickle
import pyroomacoustics as pra
import logging


def angular_error_deg(est_deg, ref_deg):
    return min(abs(est_deg - ref_deg), 360 - abs(est_deg - ref_deg))


def setup_logger(log_path):
    """ログ設定（ファイルにのみ出力）"""
    logger = logging.getLogger("DoALogger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 重複出力を防ぐためリセット

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    return logger


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

    # ログファイルのパスを自動設定
    log_path = save_path.replace(".pkl", ".log") if save_path else "doa_log_output.log"
    logger = setup_logger(log_path)

    logger.info(f"Loading: {npz_path}")
    data = np.load(npz_path)

    pred_sig = data['pred_sig']        # (N, T), complex64
    ori_sig = data['ori_sig']          # (N, T), complex64
    position_rx = data['position_rx']  # (N, 3)
    position_tx = data['position_tx']  # (N, 3)

    N = pred_sig.shape[0]
    M = 8  # マイク数
    G = N // M  # グループ数

    logger.info(f"Total samples: {N}, divided into {G} groups of {M} microphones.")

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
                logger.info(f"[{algo}] Group {g} failed: {e}")
                for key in doa_results[algo]:
                    doa_results[algo][key].append(None)

    # === ログ出力 ===
    logger.info("\n=== DoA Estimation Summary ===")
    for algo in algo_names:
        logger.info(f"\n[Algorithm: {algo}]")
        for key in ["pred_vs_gt_error", "pred_vs_true_error", "gt_vs_true_error"]:
            values = [v for v in doa_results[algo][key] if v is not None]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                logger.info(f"{key:22s} → Mean: {mean:.2f}°, Std: {std:.2f}°")
            else:
                logger.info(f"{key:22s} → No valid results")

    # 結果保存
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(doa_results, f)
        logger.info(f"\nSaved DoA results to: {save_path}")


if __name__ == "__main__":
    run_doa_on_npz(
        npz_path="/home/ach17616qc/logs/pra/Pra_param_1_1/val_result/val_iter099600.npz",
        fs=16000,
        n_fft=512,
        mic_radius=0.0365,
        algo_names=['MUSIC', 'NormMUSIC', 'SRP', 'CSSM', 'WAVES', 'TOPS', 'FRIDA'],
        save_path="/home/ach17616qc/logs/pra/Pra_param_1_1/doa_val_iter099600.pkl"
    )

