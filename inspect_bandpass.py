#!/usr/bin/env python3
# inspect_bandpass.py
# 1) 全グループの |H(f)| カーブを 1枚 (横4×縦5) に集約
#    - 各サブプロット: 薄線=8ch、太線=8ch平均
# 2) 先頭グループのみ：8ch STFT スペクトログラムを1枚に8面で保存（従来通り）

import os, argparse, math
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

def freq_axis_from_rfft_len(n_rfft: int, fs: int) -> np.ndarray:
    return np.linspace(0.0, fs/2, n_rfft)

def to_db(x, ref):
    x = np.maximum(x, 1e-12)
    ref = max(float(ref), 1e-12)
    return 20.0*np.log10(x / ref)

def load_freq_ir(npz_path, which):
    d = np.load(npz_path)
    key = "pred_sig" if which.lower() == "pred" else "ori_sig"
    if key not in d:
        raise ValueError(f"{key} not found in {npz_path}")
    H = d[key]  # (N, F), rFFT想定（0..fs/2）
    return H

def stft_spectrogram(sig, n_fft, hop, win):
    # -> (F, Tfrm), magnitude
    X = pra.transform.stft.analysis(sig, n_fft, hop, win=win)  # (Tfrm, F)
    return np.abs(X.T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="18×8chが入った単一npzのパス")
    ap.add_argument("--which", default="pred", choices=["pred","ori"])
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--group_size", type=int, default=8)

    # 1) まとめ図レイアウト & 表示
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--grid_rows", type=int, default=5)
    ap.add_argument("--ylim_db_curve", type=float, nargs=2, default=(-60, 5))

    # 2) STFT（先頭グループのみ）
    ap.add_argument("--stft_nfft", type=int, default=512)
    ap.add_argument("--stft_hop", type=int, default=256)
    ap.add_argument("--stft_db_floor", type=float, default=-60.0)

    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # ---- 入力読み込み ----
    H = load_freq_ir(args.npz, args.which)  # (N,F)
    N, F = H.shape
    if N % args.group_size != 0:
        raise SystemExit(f"N={N} が group_size={args.group_size} の倍数ではありません。")
    G = N // args.group_size

    fs = args.fs
    f = freq_axis_from_rfft_len(F, fs)

    # ---- 出力先 & 基本情報 ----
    outdir = args.outdir or os.path.join(os.path.dirname(args.npz), "bandpass_view")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.npz))[0]

    global_max = float(np.max(np.abs(H))) if np.max(np.abs(H)) > 0 else 1.0

    # ==========================================================
    # 1) 全グループまとめ図（横4×縦5に1枚）
    # ==========================================================
    ncols = args.grid_cols
    nrows = args.grid_rows
    if nrows * ncols < G:
        # グリッドが足りなければ行数を増やす
        nrows = math.ceil(G / ncols)

    # 1サブプロットの見やすさ優先で少し広め
    figsize = (3.2 * ncols, 2.6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ax_list = axes.ravel()

    for g in range(G):
        ax = ax_list[g]
        idx = slice(g * args.group_size, (g + 1) * args.group_size)  # 8ch
        Hg = H[idx]                      # (8, F)
        mag = np.abs(Hg)                 # (8, F)
        mag_mean = mag.mean(axis=0)      # (F,)
        mag_db_mean = to_db(mag_mean, global_max)

        # 薄線：各ch
        for i in range(mag.shape[0]):
            ax.plot(f, to_db(mag[i], global_max), linewidth=0.6, alpha=0.35)
        # 太線：平均
        ax.plot(f, mag_db_mean, linewidth=1.8)

        ax.set_title(f"G{g+1:02d}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(f[0], f[-1])
        ax.set_ylim(*args.ylim_db_curve)

    # 余白の軸は消す
    for k in range(G, nrows * ncols):
        fig.delaxes(ax_list[k])

    # ラベル（左端の列と下段のみ）
    for r in range(nrows):
        for c in range(ncols):
            idx_ax = r * ncols + c
            if idx_ax >= len(ax_list) or idx_ax >= G:
                continue
            ax = axes[r, c]
            if c == 0:
                ax.set_ylabel("Mag [dB]")
            if r == nrows - 1:
                ax.set_xlabel("Freq [Hz]")

    fig.suptitle(f"{stem} ({args.which}_sig)  |H(f)| per group (thin: 8ch, thick: mean)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    grid_path = os.path.join(outdir, f"{stem}_{args.which}_groups_curve_grid_{nrows}x{ncols}.png")
    fig.savefig(grid_path, dpi=300)
    plt.close(fig)

    # ==========================================================
    # 2) 先頭グループのみ：8ch STFT を 8面で（従来通り）
    # ==========================================================
    group0 = slice(0, args.group_size)
    Hg0 = H[group0]                                    # (8, F)
    ir_time = np.fft.irfft(Hg0, axis=1).astype(np.float32)  # (8, Nt)

    win = pra.windows.hann(args.stft_nfft, flag='asymmetric', length='full')
    specs = []
    max_mag = 1e-12
    for ch in range(ir_time.shape[0]):
        S = stft_spectrogram(ir_time[ch], args.stft_nfft, args.stft_hop, win)  # (Fstft, Tfrm)
        specs.append(S)
        m = float(S.max())
        if m > max_mag: max_mag = m

    f_stft = np.linspace(0, fs/2, specs[0].shape[0])
    t_stft = np.arange(specs[0].shape[1]) * (args.stft_hop / fs)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    vmin, vmax = args.stft_db_floor, 0.0
    im = None
    for ch in range(8):
        ax = axes[ch]
        SdB = to_db(specs[ch], max_mag)
        im = ax.imshow(SdB, origin="lower", aspect="auto",
                       extent=[t_stft[0], t_stft[-1] if len(t_stft) > 1 else 0,
                               f_stft[0], f_stft[-1]],
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Ch {ch+1}", fontsize=10)
        ax.grid(False)

    for i in [0,1,2,3]:
        axes[i].set_xlabel("")
    for i in [0,4]:
        axes[i].set_ylabel("Freq [Hz]")
    for i in [4,5,6,7]:
        axes[i].set_xlabel("Time [s]")

    fig.suptitle(f"{stem} ({args.which}_sig)  group 01  STFT (nfft={args.stft_nfft}, hop={args.stft_hop})", y=0.995)
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Magnitude [dB re max of group01]")
    stft_path = os.path.join(outdir, f"{stem}_{args.which}_group01_stft_8ch.png")
    fig.savefig(stft_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] curves grid :", grid_path)
    print("[OK] stft 8ch    :", stft_path)
    print("[OK] saved to    :", outdir)

if __name__ == "__main__":
    main()
