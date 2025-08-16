import os
import re
import sys
import yaml
import math
import argparse
import optuna
from avr_runner import AVR_Runner
from plot_eval import run_doa_on_npz
import pickle
import numpy as np

def update_config(config, base_start_index, trial_index=None, trial=None):
    """YAML設定ファイルを更新"""

    # === ハイパラ探索範囲 ===
    if trial:
        # 共通パラメータ
        das_reg_loss_weight = trial.suggest_float('das_reg_loss_weight', 0, 100)

        # das_reg_loss_weight > 0 なら batch_size=8 固定、それ以外は通常探索
        if das_reg_loss_weight > 0:
            batch_size = 8
        else:
            batch_size = 2 ** trial.suggest_int('batch_size', 0, 3)

        lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
        eta_min = trial.suggest_float("eta_min", lr * 1e-2, lr * 5e-1, log=True)
        n_samples = trial.suggest_int('n_samples', 40, 80)
        n_azi = trial.suggest_int('n_azi', 48, 80)
        weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
        spec_loss_weight = trial.suggest_float('spec_loss_weight', 0, 100)
        angle_loss_weight = trial.suggest_float('angle_loss_weight', 0, 100)
        time_loss_weight = trial.suggest_float('time_loss_weight', 0, 100)
        energy_loss_weight = trial.suggest_float('energy_loss_weight', 0, 100)
        multistft_loss_weight = trial.suggest_float('multistft_loss_weight', 0, 100)
        sigma_enc_neurons = 2 ** trial.suggest_int('sigma_encoder_network_n_neurons', 5, 9)
        sigma_dec_neurons = 2 ** trial.suggest_int('sigma_decoder_network_n_neurons', 5, 9)
        signal_neurons = 2 ** trial.suggest_int('signal_network_n_neurons', 7, 10)

        # channel_embed 関連
        is_embed = trial.suggest_categorical('is_embed', [True, False])
        connection_type = None
        emb_dim_enc = 0
        emb_dim_dec = 0
        emb_dim_sig = 0
        is_sigma_encoder = False
        is_sigma_decoder = False
        is_signal_network = False

        if is_embed:
            connection_type = trial.suggest_categorical('channel_embed_connection_type', ['add', 'concat'])

            if connection_type == 'concat':
                is_sigma_encoder = trial.suggest_categorical('is_sigma_encoder', [True, False])
                is_sigma_decoder = trial.suggest_categorical('is_sigma_decoder', [True, False])
                is_signal_network = trial.suggest_categorical('is_signal_network', [True, False])
                if is_sigma_encoder:
                    emb_dim_enc = 2 ** trial.suggest_int('emb_dim_sigma_encoder', 5, 8)
                if is_sigma_decoder:
                    emb_dim_dec = 2 ** trial.suggest_int('emb_dim_sigma_decoder', 5, 8)
                if is_signal_network:
                    emb_dim_sig = 2 ** trial.suggest_int('emb_dim_signal_network', 5, 8)
            elif connection_type == 'add':
                is_sigma_encoder = trial.suggest_categorical('is_sigma_encoder', [True, False])
                is_sigma_decoder = trial.suggest_categorical('is_sigma_decoder', [True, False])
                is_signal_network = trial.suggest_categorical('is_signal_network', [True, False])
    else:
        # trial指定なし → configからそのまま
        das_reg_loss_weight = config['train'].get('das_reg_loss_weight', 0)
        batch_size = config['train']['batch_size']
        lr = config['train']['lr']
        eta_min = config['train']['eta_min']
        n_samples = config['render']['n_samples']
        n_azi = config['render']['n_azi']
        weight_decay = config['train']['weight_decay']
        spec_loss_weight = config['train']['spec_loss_weight']
        angle_loss_weight = config['train']['angle_loss_weight']
        time_loss_weight = config['train']['time_loss_weight']
        energy_loss_weight = config['train']['energy_loss_weight']
        multistft_loss_weight = config['train']['multistft_loss_weight']
        sigma_enc_neurons = config['model']['sigma_encoder_network']['n_neurons']
        sigma_dec_neurons = config['model']['sigma_decoder_network']['n_neurons']
        signal_neurons = config['model']['signal_network']['n_neurons']

        channel_embed_cfg = config['model'].get('channel_embed', {})
        is_embed = channel_embed_cfg.get('is_embed', False)
        connection_type = channel_embed_cfg.get('connection_type', None)
        is_sigma_encoder = channel_embed_cfg.get('is_sigma_encoder', False)
        is_sigma_decoder = channel_embed_cfg.get('is_sigma_decoder', False)
        is_signal_network = channel_embed_cfg.get('is_signal_network', False)
        emb_dim_enc = channel_embed_cfg.get('emb_dim_sigma_encoder', 0)
        emb_dim_dec = channel_embed_cfg.get('emb_dim_sigma_decoder', 0)
        emb_dim_sig = channel_embed_cfg.get('emb_dim_signal_network', 0)

    # === バッチサイズスケーリング（切り上げ）===
    scale = batch_size / config['train']['batch_size']
    config['train']['batch_size'] = batch_size
    config['train']['T_max'] = math.ceil(config['train']['T_max'] / scale)
    config['train']['total_iterations'] = math.ceil(config['train']['total_iterations'] / scale)
    config['train']['save_freq'] = math.ceil(config['train']['save_freq'] / scale)
    config['train']['val_freq'] = math.ceil(config['train']['val_freq'] / scale)

    # === その他パラメータ更新 ===
    config['train']['lr'] = lr
    config['train']['eta_min'] = eta_min
    config['train']['das_reg_loss_weight'] = das_reg_loss_weight
    config['render']['n_samples'] = n_samples
    config['render']['n_azi'] = n_azi
    config['train']['weight_decay'] = weight_decay
    config['train']['spec_loss_weight'] = spec_loss_weight
    config['train']['angle_loss_weight'] = angle_loss_weight
    config['train']['time_loss_weight'] = time_loss_weight
    config['train']['energy_loss_weight'] = energy_loss_weight
    config['train']['multistft_loss_weight'] = multistft_loss_weight

    config['model']['sigma_encoder_network']['n_neurons'] = sigma_enc_neurons
    config['model']['sigma_decoder_network']['n_neurons'] = sigma_dec_neurons
    config['model']['signal_network']['n_neurons'] = signal_neurons

    # channel_embed関連の設定
    if 'channel_embed' not in config['model']:
        config['model']['channel_embed'] = {}

    config['model']['channel_embed']['is_embed'] = is_embed
    config['model']['channel_embed']['connection_type'] = connection_type
    config['model']['channel_embed']['is_sigma_encoder'] = is_sigma_encoder
    config['model']['channel_embed']['is_sigma_decoder'] = is_sigma_decoder
    config['model']['channel_embed']['is_signal_network'] = is_signal_network
    config['model']['channel_embed']['emb_dim_sigma_encoder'] = emb_dim_enc
    config['model']['channel_embed']['emb_dim_sigma_decoder'] = emb_dim_dec
    config['model']['channel_embed']['emb_dim_signal_network'] = emb_dim_sig

    # === expname 更新 ===
    trial_num = base_start_index if trial_index is None else base_start_index + trial_index
    base_name = config['path']['expname']
    new_expname = re.sub(r'param_\d+_1', f'param_{trial_num}_1', base_name)
    if new_expname == base_name:
        new_expname = f"{base_name.split('param_')[0]}param_{trial_num}_1"
    config['path']['expname'] = new_expname

    return config

def run_training_and_doa(config, dataset_dir):
    """学習とDoA評価"""
    batch_size = config.get("train", {}).get("batch_size", 4)
    logdir = os.path.join(config['path']['logdir'], config['path']['expname'])
    os.makedirs(logdir, exist_ok=True)

    img_train_dir = os.path.join(logdir, 'img_train')
    os.makedirs(img_train_dir, exist_ok=True)
    img_test_dir = os.path.join(logdir, 'img_test')
    os.makedirs(img_test_dir, exist_ok=True)

    runner = AVR_Runner(mode='train', dataset_dir=dataset_dir, batchsize=batch_size, **config)
    runner.train()

    npz_dir = os.path.join(logdir, "val_result")
    doa_save_dir = os.path.join(logdir, "doa_results")
    os.makedirs(doa_save_dir, exist_ok=True)

    # val_iter 限定で検索
    val_npz_files = sorted([
        os.path.join(npz_dir, f)
        for f in os.listdir(npz_dir)
        if re.match(r"val_iter\d+\.npz", f)
    ], key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0]))

    doa_errors = []
    for npz_path in val_npz_files:
        filename = os.path.splitext(os.path.basename(npz_path))[0] + ".pkl"
        doa_pkl = os.path.join(doa_save_dir, filename)

        if not os.path.exists(doa_pkl):
            run_doa_on_npz(
                npz_path=npz_path,
                fs=config['render']['fs'],
                save_path=doa_pkl
            )
        with open(doa_pkl, "rb") as f:
            doa_results = pickle.load(f)
        errors = [e for e in doa_results["NormMUSIC"]["pred_vs_gt_error"] if e is not None]
        if errors:
            doa_errors.append(np.mean(errors))

    return min(doa_errors) if doa_errors else 999

def objective(trial):
    global base_config, args
    trial_index = trial.number
    config = update_config(yaml.load(yaml.dump(base_config), Loader=yaml.FullLoader),
                           args.start_index, trial_index, trial)

    trial_dir = os.path.join(config['path']['logdir'], config['path']['expname'])
    os.makedirs(trial_dir, exist_ok=True)
    yaml_path = os.path.join(trial_dir, f"avr_conf_trial_{trial_index}.yml")
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    best_doa_error = run_training_and_doa(config, args.dataset_dir)
    return best_doa_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--start_index', type=int, required=True)
    parser.add_argument('--n_trials', type=int, default=50)
    # 追加: スタディ名と保存先（SQLite）
    parser.add_argument('--study_name', type=str, default='avr_optuna_study',
                        help='Optuna study name (used for resume)')
    parser.add_argument('--storage', type=str, default='sqlite:///./optuna_avr.db',
                        help='Optuna storage URL; SQLite file will be created if missing.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        base_config = yaml.load(file, Loader=yaml.FullLoader)

    # ここだけ変更: storage を指定し、load_if_exists=True で再開可能に
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    print("Study name:", study.study_name)
    print("Storage:", args.storage)
