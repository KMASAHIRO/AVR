import os
import re
import glob
import argparse
import yaml
from plot_eval import run_doa_on_npz, run_delay_and_sum_on_npz, plot_loss_and_doa_over_epochs

def find_tensorboard_event_file(tensorboard_logdir: str, relative_subpath: str, expname: str):
    """
    tensorboard_logdir/relative_subpath/expname/**/events.out.tfevents.*
    という構成からイベントファイルを検索
    """
    exp_dir = os.path.join(tensorboard_logdir, relative_subpath, expname)
    pattern = os.path.join(exp_dir, '**', 'events.out.tfevents.*')
    event_files = glob.glob(pattern, recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {exp_dir}")
    return event_files[0]

def execute_from_yaml_config(args):
    # === YAML読み込み ===
    with open(args.config, 'r') as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)

    path_cfg = kwargs.get("path", {})
    expname = path_cfg["expname"]
    logdir = path_cfg["logdir"]  # e.g., logs/real_exp/

    # === 2階層目以降のパス抽出（logs/real_exp/ → real_exp） ===
    logdir_subpath = "/".join(logdir.split("/")[1:])  # "real_exp" など

    # === TensorBoardログパスを取得 ===
    tensorboard_log_path = find_tensorboard_event_file(args.tensorboard_logdir, logdir_subpath, expname)

    # === DoAのnpz/pkl/outputパスを設定（logdir/expname 以下） ===
    base_path = os.path.join(logdir, expname)
    doa_npz_dir = os.path.join(base_path, "val_result")
    doa_save_dir = os.path.join(base_path, "doa_results")
    output_path_default = os.path.join(base_path, "loss_and_doa_plot.png")
    
    train_cfg = kwargs.get("train", {})
    das_reg_weight = train_cfg.get("das_reg_loss_weight", 0.0)
    das_ce_weight = train_cfg.get("das_ce_loss_weight", 0.0)

    # === Beamformingの結果保存先 ===
    beamform_save_dir = os.path.join(base_path, "beamform_results")
    output_path_beamform = os.path.join(base_path, "loss_and_doa_and_beamform_plot.png")

    # === 実行 ===
    if das_reg_weight > 0 or das_ce_weight > 0:
        # DAS 損失が有効な場合 → beamformingもプロット
        plot_loss_and_doa_over_epochs(
            log_path=tensorboard_log_path,
            doa_npz_dir=doa_npz_dir,
            doa_save_dir=doa_save_dir,
            algo_name="NormMUSIC",
            error_key="pred_vs_gt_error",
            output_path=output_path_beamform,
            run_doa_func=run_doa_on_npz,
            run_beamform_func=run_delay_and_sum_on_npz,
            beamform_save_dir=beamform_save_dir,
        )
    else:
        # 通常のDoAプロットのみ
        plot_loss_and_doa_over_epochs(
            log_path=tensorboard_log_path,
            doa_npz_dir=doa_npz_dir,
            doa_save_dir=doa_save_dir,
            algo_name="NormMUSIC",
            error_key="pred_vs_gt_error",
            output_path=output_path_default,
            run_doa_func=run_doa_on_npz
        )

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Loss and DoA Error from YAML Config")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--tensorboard_logdir', default="tensorboard_logs", help='Base directory of TensorBoard logs')

    args = parser.parse_args()
    execute_from_yaml_config(args)
