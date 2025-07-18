import os
import re
import yaml
from pathlib import Path
from copy import deepcopy

def generate_param_variants(base_config_dir: str, param_dict: dict):
    base_path = Path(base_config_dir)
    last_dir = base_path.name
    capitalized = last_dir.capitalize()
    base_file = base_path / f'avr_{last_dir}_1.yml'

    if not base_file.exists():
        raise FileNotFoundError(f"Base config file {base_file} not found")

    with open(base_file, 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    base_expname = base_config['path']['expname']
    match = re.search(rf"{capitalized}_param_(\d+)", base_expname)
    if not match:
        raise ValueError("expname format invalid")
    base_idx = int(match.group(1))
    file_count = 0

    for section, params in param_dict.items():
        if section == "model":
            # 特例：model のみ2階層処理
            for key1, val1 in params.items():
                if isinstance(val1, dict):
                    for key2, values in val1.items():
                        for v in values:
                            new_config = deepcopy(base_config)
                            new_config["model"][key1][key2] = v

                            file_count += 1
                            new_idx = base_idx + file_count
                            new_expname = re.sub(
                                rf"{capitalized}_param_\d+",
                                f"{capitalized}_param_{new_idx}",
                                base_expname
                            )
                            new_config['path']['expname'] = new_expname

                            output_file = base_path / f'avr_{last_dir}_{new_idx}.yml'
                            with open(output_file, 'w') as f:
                                yaml.dump(new_config, f, sort_keys=False)

                            print(f"✅ Generated: {output_file}")
                else:
                    # model の key1 が list（1階層）ならそのまま処理
                    for v in val1:
                        new_config = deepcopy(base_config)
                        new_config["model"][key1] = v

                        file_count += 1
                        new_idx = base_idx + file_count
                        new_expname = re.sub(
                            rf"{capitalized}_param_\d+",
                            f"{capitalized}_param_{new_idx}",
                            base_expname
                        )
                        new_config['path']['expname'] = new_expname

                        output_file = base_path / f'avr_{last_dir}_{new_idx}.yml'
                        with open(output_file, 'w') as f:
                            yaml.dump(new_config, f, sort_keys=False)

                        print(f"✅ Generated: {output_file}")
        else:
            # 通常セクション（1階層）
            for key, values in params.items():
                for v in values:
                    new_config = deepcopy(base_config)
                    new_config[section][key] = v

                    file_count += 1
                    new_idx = base_idx + file_count
                    new_expname = re.sub(
                        rf"{capitalized}_param_\d+",
                        f"{capitalized}_param_{new_idx}",
                        base_expname
                    )
                    new_config['path']['expname'] = new_expname

                    output_file = base_path / f'avr_{last_dir}_{new_idx}.yml'
                    with open(output_file, 'w') as f:
                        yaml.dump(new_config, f, sort_keys=False)

                    print(f"✅ Generated: {output_file}")

    print(f"✔️ Total YAML files generated: {file_count}")

if __name__ =="__main__":
    param_dict = {
        "train": {
            "batch_size": [2, 8],
            "weight_decay": [1e-5, 1e-4],
            "spec_loss_weight": [1, 4],
            "amplitude_loss_weight": [2, 8],
            "angle_loss_weight": [0.5, 2],
            "time_loss_weight": [25, 100],
            "energy_loss_weight": [0.5, 2],
            "multistft_loss_weight": [0.5, 2]
        },
        "render": {
            "n_samples": [48, 80],
            "n_azi": [48, 80],
            "n_ele": [16, 48]
        },
        "model": {
            "sigma_encoder_network": {
                "n_neurons": [64, 256]
            },
            "sigma_decoder_network": {
                "n_neurons": [64, 256]
            },
            "signal_network": {
                "n_neurons": [256, 1024]
            }
        }
    }
    
    generate_param_variants("config_files/pra", param_dict)
    generate_param_variants("config_files/pra_ch_emb", param_dict)
    generate_param_variants("config_files/real_env", param_dict)
    generate_param_variants("config_files/real_env_ch_emb", param_dict)
    generate_param_variants("config_files/real_exp", param_dict)
    generate_param_variants("config_files/real_exp_ch_emb", param_dict)

