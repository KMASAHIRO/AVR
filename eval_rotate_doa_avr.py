# eval_rotate_doa_avr.py
import os, math, argparse, yaml
import numpy as np
import torch
import pyroomacoustics as pra

from datasets_loader import WaveLoader
from model import AVRModel, AVRModel_complex
from renderer import AVRRender

# ---------- utils ----------
def load_yaml(path):
    with open(os.path.expanduser(path), "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def rotate_about_xy(center_xy, radius, deg):
    th = math.radians(deg)
    cx, cy = center_xy
    return np.array([cx + radius * math.cos(th), cy + radius * math.sin(th)], dtype=float)

def in_bounds_3d(p, min_xyz, max_xyz):
    return (min_xyz[0] <= p[0] <= max_xyz[0]) and (min_xyz[1] <= p[1] <= max_xyz[1]) and (min_xyz[2] <= p[2] <= max_xyz[2])

def angular_error_deg(a, b):
    a = np.asarray(a) % 360.0
    b = np.asarray(b) % 360.0
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)

def build_renderer_from_cfg(cfg, device):
    dataset_type = cfg['path']['dataset_type']
    kwargs_network = cfg['model']
    kwargs_render = cfg['render']

    if dataset_type in ['MeshRIR', 'Simu', 'Real_env']:
        net = AVRModel(kwargs_network)
    elif dataset_type == 'RAF':
        net = AVRModel_complex(kwargs_network)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    renderer = AVRRender(networks_fn=net, **kwargs_render)
    if torch.cuda.device_count() > 1:
        renderer = torch.nn.DataParallel(renderer)
    renderer = renderer.to(device).eval()
    return renderer, kwargs_render['fs']

def load_checkpoint(renderer, ckpt_path, device):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("[WARN] ckpt not provided; using random weights.")
        return renderer
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location=device)
    state_dict = ckpt.get('audionerf_network_state_dict', ckpt)
    try:
        renderer.load_state_dict(state_dict)
    except Exception:
        renderer.module.load_state_dict(state_dict)
    print("[INFO] Checkpoint loaded.")
    return renderer

# ---------- main ----------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--dataset_dir', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default=None)

    # rotation & doa
    ap.add_argument('--deg_step', type=float, default=10.0)
    ap.add_argument('--array_radius', type=float, default=0.0365)
    ap.add_argument('--nfft', type=int, default=512)

    # bounds (fixed as specified)
    ap.add_argument('--min_xyz', type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument('--max_xyz', type=float, nargs=3, default=[6.11, 8.807, 2.7])

    # grouping
    ap.add_argument('--group_size', type=int, default=8)  # 8ch rigid rotation assumed
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # renderer + ckpt
    renderer, fs = build_renderer_from_cfg(cfg, device)
    renderer = load_checkpoint(renderer, args.ckpt, device)
    renderer.eval()

    # dataset (eval=True)
    ds_type = cfg['path']['dataset_type']
    seq_len = cfg['model']['signal_output_dim']
    test_set = WaveLoader(base_folder=args.dataset_dir, dataset_type=ds_type, eval=True, seq_len=seq_len, fs=fs)

    out_dir = args.out_dir or os.path.join(cfg['path']['logdir'], cfg['path']['expname'], 'rotate_eval_avr')
    os.makedirs(out_dir, exist_ok=True)

    min_xyz = np.array(args.min_xyz, dtype=np.float32)
    max_xyz = np.array(args.max_xyz, dtype=np.float32)
    delta_list = [k * args.deg_step for k in range(int(360 // args.deg_step))]

    def render_freq(rx_xyz, tx_xyz, ch_idx=None):
        rx = torch.from_numpy(rx_xyz).float().to(device).view(1, -1)
        tx = torch.from_numpy(tx_xyz).float().to(device).view(1, -1)
        ch_tensor = None if (ch_idx is None or ch_idx == -1) else torch.tensor([ch_idx], device=device)
        pred = renderer(rx, tx, ch_idx=ch_tensor)         # (...,2)
        pred = pred[...,0] + 1j * pred[...,1]             # (1, F)  # F = rFFT 長 = seq_len//2 + 1
        return pred[0].detach().cpu().numpy().astype(np.complex64)  # (F,)

    # summary holders
    summary_lines = ["unit_id,used_rotations,mean_err_deg\n"]
    all_pred_deg, all_true_deg = [], []

    # flat outputs to match run_doa_on_npz (N = M * N_rot)
    flat_pred_spec = []   # [(T_fft,), ...]
    flat_pos_rx = []      # [(3,), ...]
    flat_pos_tx = []      # [(3,), ...]

    # group rigid rotation
    N = len(test_set)
    gid, start = 0, 0
    while start < N:
        idxs = list(range(start, min(start + args.group_size, N)))
        start += args.group_size
        if len(idxs) < args.group_size:
            break

        tx_list, rx_list, ch_list = [], [], []
        for k in idxs:
            _, pos_rx, pos_tx, ch_idx = test_set[k]
            tx_list.append(pos_tx.numpy().astype(float))
            rx_list.append(pos_rx.numpy().astype(float))
            ch_list.append(int(ch_idx.item()) if hasattr(ch_idx, 'item') else -1)

        tx0 = tx_list[0]
        tx_xy = tx0[:2]

        radii, theta0_list, z_list = [], [], []
        for rxyz in rx_list:
            rxy = rxyz[:2]
            theta0_k = (math.degrees(math.atan2(rxy[1]-tx_xy[1], rxy[0]-tx_xy[0])) % 360.0)
            theta0_list.append(theta0_k)
            radii.append(float(np.linalg.norm(rxy - tx_xy)))
            z_list.append(rxyz[2])

        used_angles, pred_deg_list, true_deg_list = [], [], []

        for d in delta_list:
            angs_k = [ (theta0_list[k] + d) % 360.0 for k in range(args.group_size) ]
            rx_rot_xyz_list = []
            for k in range(args.group_size):
                rx_rot_xy = rotate_about_xy(tx_xy, radii[k], angs_k[k])
                rx_rot_xyz = np.array([rx_rot_xy[0], rx_rot_xy[1], z_list[k]], dtype=np.float32)
                rx_rot_xyz_list.append(rx_rot_xyz)

            if not all(in_bounds_3d(p, min_xyz, max_xyz) for p in rx_rot_xyz_list):
                continue
            used_angles.append(d)

            pred_freq_stack = []
            for k in range(args.group_size):
                pred_f = render_freq(rx_rot_xyz_list[k], tx0, ch_idx=ch_list[k])  # (F,)
                pred_freq_stack.append(pred_f)
            pred_freq_stack = np.stack(pred_freq_stack, axis=0).astype(np.complex64)  # (M, F)

            # STFT/DOA 用に時間波形が必要な時だけ irfft で戻す（これはOK）
            time_for_stft = np.fft.irfft(pred_freq_stack, n=seq_len, axis=-1).astype(np.float32)  # (M, T)
            
            # ループ内で group_size=8 チャンネル分を処理
            for k in range(args.group_size):
                flat_pred_spec.append(pred_freq_stack[k])   # (F,) 各チャンネルの周波数スペクトル
                flat_pos_rx.append(rx_rot_xyz_list[k])      # (3,)
                flat_pos_tx.append(tx0)                     # (3,)

            # DoA just to record angles (STFT like your code)
            X_pred = np.array([pra.transform.stft.analysis(sig, args.nfft, args.nfft // 2)
                               for sig in time_for_stft])  # (M, F, Tfrm)
            X_pred = np.transpose(X_pred, (0, 2, 1))  # (M, Tfrm, F)

            mic_center = np.mean(np.stack(rx_rot_xyz_list, axis=0)[:, :2], axis=0)
            mic = pra.beamforming.circular_2D_array(center=mic_center, M=args.group_size,
                                                    radius=args.array_radius, phi0=np.pi/2)
            doa = pra.doa.algorithms["NormMUSIC"](mic, fs=fs, nfft=args.nfft)
            doa.locate_sources(X_pred)
            pred_deg_list.append(np.int16(int(np.argmax(doa.grid.values) % 360)))

            dx, dy = tx0[0] - mic_center[0], tx0[1] - mic_center[1]
            true_deg = (math.degrees(math.atan2(dy, dx)) % 360.0)
            true_deg_list.append(np.int16(int(true_deg)))

        if used_angles:
            pred_arr = np.array(pred_deg_list, dtype=np.int16)
            true_arr = np.array(true_deg_list, dtype=np.int16)
            err = angular_error_deg(pred_arr, true_arr)
            summary_lines.append(f"{gid},{len(used_angles)},{float(err.mean()):.4f}\n")
            all_pred_deg.append(pred_arr); all_true_deg.append(true_arr)
        else:
            summary_lines.append(f"{gid},0,NaN\n")
        gid += 1

    # 保存（フラット）
    flat_pred_spec = np.stack(flat_pred_spec, axis=0).astype(np.complex64)  # (N, F)
    flat_pos_rx = np.stack(flat_pos_rx, axis=0).astype(np.float32)          # (N, 3)
    flat_pos_tx = np.stack(flat_pos_tx, axis=0).astype(np.float32)          # (N, 3)

    # 各グループの pred_deg / true_deg を結合
    if len(all_pred_deg) > 0:
        flat_pred_deg = np.concatenate(all_pred_deg).astype(np.int16)
        flat_true_deg = np.concatenate(all_true_deg).astype(np.int16)
    else:
        flat_pred_deg = np.array([], dtype=np.int16)
        flat_true_deg = np.array([], dtype=np.int16)

    save_path = os.path.join(out_dir, f"val_rotate_pred.npz")
    np.savez_compressed(
        save_path,
        pred_sig=flat_pred_spec,      # (N, F) complex64
        position_rx=flat_pos_rx,      # (N, 3)
        position_tx=flat_pos_tx,      # (N, 3)
        pred_deg=flat_pred_deg,       # (N_rot,) deg
        true_deg=flat_true_deg,       # (N_rot,) deg
        fs=np.int32(fs),
        n_fft=np.int32(args.nfft),
        mic_radius=np.float32(args.array_radius),
        group_size=np.int32(args.group_size),
        deg_step=np.float32(args.deg_step)
    )
    
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.writelines(summary_lines)

    if len(all_pred_deg):
        pred_all = np.concatenate(all_pred_deg); true_all = np.concatenate(all_true_deg)
        overall = float(angular_error_deg(pred_all, true_all).mean())
        with open(os.path.join(out_dir, "overall.txt"), "w") as f:
            f.write(f"mean_angular_error_deg={overall:.4f}\n")
        print(f"[DONE] overall mean angular error = {overall:.4f}°")
    else:
        print("[DONE] No usable rotations.")

if __name__ == "__main__":
    main()
