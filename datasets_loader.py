import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import glob
import librosa
import math
    
class WaveLoader(Dataset):
    def __init__(self, base_folder, dataset_type='MeshRIR', eval=False, seq_len=2048, fs=16000):
        """DataLoader initializations, can load three different sets together

        Parameters
        ----------
        base_folder : string
            path to dataset
        dataset_type : str, optional
            dataset_type, by default 'MeshRIR', can be selected 'RAF', 'Simu'
        eval : bool, optional
            flag to determine training or testing set
        seq_len : int, optional
            length of the prediction audio
        fs : int, optional
            sampling rate of the audio, by default is 16000
        """
        
        self.wave_chunks = []
        self.positions_rx = []
        self.positions_tx = []
        self.rotations_tx = []
        self.ch_idx_list = []

        self.wave_max = float('-inf')
        self.wave_min = float('inf')
        self.position_max = np.array([float('-inf'), float('-inf'), float('-inf')])
        self.position_min = np.array([float('inf'), float('inf'), float('inf')])

        self.dataset_type = dataset_type
        self.eval = eval

        # load three different datasets seperately
        if dataset_type == 'MeshRIR':
            self.load_mesh_rir(base_folder, eval, seq_len, fs)
        elif dataset_type == 'RAF':
            self.load_raf(base_folder, eval, seq_len, fs)
        elif dataset_type == 'Simu':
            self.load_simu(base_folder, eval, seq_len, fs)
        elif dataset_type == 'Real_env':
            self.load_real_env(base_folder, eval, seq_len, fs)
        else:
            raise ValueError("Unsupported dataset type")

        # Convert lists to tensors for faster processing in __getitem__
        self.wave_chunks = torch.tensor(np.array(self.wave_chunks), dtype=torch.complex64)
        self.positions_rx = torch.tensor(np.array(self.positions_rx), dtype=torch.float32)
        self.positions_tx = torch.tensor(np.array(self.positions_tx), dtype=torch.float32)
        if self.rotations_tx:
            self.rotations_tx = torch.tensor(np.array(self.rotations_tx), dtype=torch.float32)

    def load_mesh_rir(self, base_folder, eval, seq_len, fs=24000):
        """ Load MeshRIR datasets
        """
        down_sample_rate = 48000 // fs
        self.default_st_idx = int(9100 / down_sample_rate)

        if eval:
            wave_folder = os.path.join(base_folder, 'test')
        else:
            wave_folder = os.path.join(base_folder, 'train')

        filenames = [f for f in os.listdir(wave_folder) if f.endswith('.npy')]
        filenames.sort()

        rx_pos = np.load(os.path.join(base_folder, 'pos_mic.npy'))
        tx_pos = np.load(os.path.join(base_folder, 'pos_src.npy'))[0]

        for filename in filenames:
            audio_data = np.load(os.path.join(wave_folder, filename))[0,::down_sample_rate] # first resample the IR data
            audio_data = audio_data[self.default_st_idx:self.default_st_idx+seq_len] # index the IR data.
            wave_data = np.fft.rfft(audio_data)

            file_ind = int(filename.split('_')[1].split('.')[0])
            position_rx = rx_pos[file_ind]
            position_tx = tx_pos

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(wave_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)

    def load_simu(self, base_folder, eval, seq_len, fs):
        """ Load simulation datasets
        """
        filenames = [f for f in os.listdir(base_folder) if f.endswith('.npz')]
        filenames.sort()

        if eval:
            filenames = filenames[int(0.9 * len(filenames)):]  # testing
        else:
            filenames = filenames[:int(0.9 * len(filenames))]  # training

        for filename in filenames:
            meta_data = np.load(os.path.join(base_folder, filename))
            audio_data = meta_data['ir'][:seq_len]
            wave_data = np.fft.rfft(audio_data)

            position_rx = meta_data['position_rx']
            position_tx = meta_data['position_tx']

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(wave_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)
    
    def load_real_env(self, base_folder, eval, seq_len, fs):
        """ Load simulation datasets for AVR using predefined split
        """
        # split.pkl を読み込む
        split_path = os.path.join(base_folder, "train_test_split.pkl")
        with open(split_path, "rb") as f:
            split = pickle.load(f)

        # eval=True なら test、False なら train を使う
        file_list = split["test"] if eval else split["train"]

        for file_path in file_list:
            # 絶対パスまたは相対パスを調整
            if not os.path.isabs(file_path):
                file_path = os.path.join(base_folder, file_path)

            # npzファイルの読み込み
            meta_data = np.load(file_path)
            audio_data = meta_data['ir'][:seq_len]
            wave_data = np.fft.rfft(audio_data)

            position_rx = meta_data['position_rx']
            position_tx = meta_data['position_tx']

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(wave_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)

            if "ch_idx" in meta_data:
                self.ch_idx_list.append(meta_data["ch_idx"].item())

    def load_raf(self, base_folder, eval, seq_len, fs):
        """ Load RAF datasets
        """
        folderpaths = glob.glob(f"{base_folder}/*")
        folderpaths.sort()

        if eval:
            folderpaths = glob.glob(f"{base_folder}/test/*")
        else:
            folderpaths = glob.glob(f"{base_folder}/train/*")
        folderpaths.sort()

        for folderpath in folderpaths:
            rir_path = os.path.join(folderpath, "rir.wav")
            audio_data, _ = librosa.load(rir_path, sr=None, mono=True)
            audio_data = audio_data[:seq_len * int(48000 / fs):int(48000 / fs)]
            wave_data = np.fft.rfft(audio_data)

            position_rx = self.load_position(os.path.join(folderpath, "rx_pos.txt"))
            position_tx, rotation_tx = self.load_tx_info(os.path.join(folderpath, "tx_pos.txt"))

            self.update_min_max(audio_data, position_rx)

            self.wave_chunks.append(wave_data)
            self.positions_rx.append(position_rx)
            self.positions_tx.append(position_tx)
            self.rotations_tx.append(rotation_tx)

    def load_position(self, file_path):
        position = []
        with open(file_path, 'r') as file:
            for line in file:
                position.extend([float(num) for num in line.split(',')])
        return np.array(position)[[0, 2, 1]]

    def load_tx_info(self, file_path):
        tx_info = []
        with open(file_path, 'r') as file:
            for line in file:
                tx_info.extend([float(num) for num in line.split(',')])
        tx_info = np.array(tx_info)
        rotation_tx = tx_info[:4]
        rotation_tx = quaternion_to_direction_vector(rotation_tx)
        position_tx = np.array(tx_info[4:])[[0, 2, 1]]
        return position_tx, rotation_tx

    def update_min_max(self, audio_data, position_rx):
        self.wave_max = max(self.wave_max, audio_data.max())
        self.wave_min = min(self.wave_min, audio_data.min())
        self.position_max = np.maximum(self.position_max, position_rx)
        self.position_min = np.minimum(self.position_min, position_rx)

    def __len__(self):
        return len(self.wave_chunks)

    def __getitem__(self, idx):
        wave_signal = self.wave_chunks[idx]
        position_rx = self.positions_rx[idx]
        position_tx = self.positions_tx[idx]
        ch_idx = self.ch_idx_list[idx] if len(self.ch_idx_list) > 0 else None
        
        if not self.eval and self.dataset_type == 'RAF':
            position_rx = position_rx + torch.randn_like(position_rx) * 0.1
            position_tx = position_tx + torch.randn_like(position_tx) * 0.1
        
        if self.dataset_type == 'RAF':
            rotation_tx = self.rotations_tx[idx]
            return wave_signal, position_rx, position_tx, rotation_tx, ch_idx
        else: 
            return wave_signal, position_rx, position_tx, ch_idx


def quaternion_to_direction_vector(q):
    """Convert a quaternion to direction vectors in Cartesian coordinates

    Parameters
    ----------
    q : Quaternion, given as a Tensor [x, y, z, w].

    Returnsdata
    -------
    Direction vectors as pts_x, pts_y, pts_z
    """

    x, y, z, w = q

    # Convert quaternion to forward direction vector
    fwd_x = 2 * (x*z + w*y)
    fwd_y = 2 * (y*z - w*x)
    fwd_z = 1 - 2 * (x*x + y*y)

    # Normalize the vector (in case it's not exactly 1 due to numerical precision)
    norm = math.sqrt(fwd_x**2 + 0**2 + fwd_z**2)
    
    return np.array([-fwd_x / norm, -fwd_z / norm, 0])