"""PIOWave training and testing
"""
import os
import argparse
from shutil import copyfile
import yaml
from tqdm import tqdm
from datetime import datetime
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets_loader import WaveLoader

from model import AVRModel, AVRModel_complex
from renderer import  AVRRender
from utils.metric import metric_cal
from utils.logger import logger_config, log_inference_figure, plot_and_save_figure
from utils.criterion import Criterion

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.parallel

def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"[{tag}] GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Max Reserved: {max_reserved:.2f}MB")


def setup_distributed():
    # 必要な環境変数が設定されているか確認
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise ValueError("Error: Missing environment variables RANK or WORLD_SIZE. Run with torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())

class AVR_Runner():
    def __init__(self, mode, dataset_dir, batchsize, **kwargs) -> None:
        ## Seperate each settings
        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['model']
        kwargs_train = kwargs['train']
        self.kwargs_train = kwargs['train']

        ## Path settings
        self.expname = kwargs_path['expname']
        self.dataset_type = kwargs_path['dataset_type']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='avr')
        self.logger.info("expname:%s, data type:%s, logdir:%s", self.expname, self.dataset_type, self.logdir)

        ## tensorboard writer
        if mode == 'train':
            log_prefix = f'tensorboard_logs/{(self.logdir).split("/")[1]}/{self.expname}'
            os.makedirs(log_prefix, exist_ok=True)
            self.writer = SummaryWriter(log_dir=f'{log_prefix}/{datetime.now().strftime("%m%d-%H%M%S")}')

        # network and renderer
        self.fs = kwargs['render']['fs']

        if self.dataset_type == 'MeshRIR' or self.dataset_type == 'Simu':
            audionerf = AVRModel(kwargs_network) # network
        elif self.dataset_type == 'RAF':
            audionerf = AVRModel_complex(kwargs_network) # network

        self.renderer = AVRRender(networks_fn=audionerf, **kwargs_render) # renderer

        # multi gpu
        if torch.cuda.device_count() > 1:
            # DDP の初期化
            #dist.init_process_group(backend="nccl")
            # DDP 初期化
            setup_distributed()

            self.local_rank = int(os.environ["LOCAL_RANK"])  # 各プロセスのローカルランクを取得
            self.device = torch.device(f"cuda:{self.local_rank}")  # 各プロセスの GPU を指定
            torch.cuda.set_device(self.device)

            self.renderer = self.renderer.to(self.device)

            self.renderer = DDP(self.renderer, device_ids=None, find_unused_parameters=True)
        else:
            self.renderer = self.renderer.cuda()

        ## Optimization
        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr=float(self.kwargs_train['lr']),
                                          weight_decay=float(self.kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))

        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),
                                                                        last_epoch=-1)
        
        ## Print total number of parameters
        params = list(self.renderer.parameters())
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        ## Train Settings
        self.current_iteration = 1
        if kwargs_train['load_ckpt']:
            self.load_checkpoints()
        self.batch_size = batchsize
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']  
        self.val_freq = kwargs_train['val_freq']

        ## dataloader
        self.train_set = WaveLoader(base_folder=dataset_dir, dataset_type=self.dataset_type, eval=False, seq_len=kwargs_network['signal_output_dim'], fs=kwargs_render['fs'])
        self.test_set = WaveLoader(base_folder=dataset_dir, dataset_type=self.dataset_type, eval=True, seq_len=kwargs_network['signal_output_dim'], fs=kwargs_render['fs'])
        self.train_set_show = WaveLoader(base_folder=dataset_dir, dataset_type=self.dataset_type, eval=False, seq_len=kwargs_network['signal_output_dim'], fs=kwargs_render['fs'])

        # DataLoader に DistributedSampler を適用
        self.train_sampler = DistributedSampler(self.train_set, shuffle=True)
        self.test_sampler = DistributedSampler(self.test_set, shuffle=False)
        self.train_sampler_show = DistributedSampler(self.train_set_show, shuffle=False)

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=4)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, sampler=self.test_sampler, num_workers=4)
        self.train_iter_show = DataLoader(self.train_set_show, batch_size=self.batch_size, sampler=self.train_sampler_show, num_workers=4)


        self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))

        # loss settings
        self.criterion = Criterion(kwargs_train)


    def load_checkpoints(self):
        """load previous checkpoints
        """
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            try: self.renderer.load_state_dict(ckpt['audionerf_network_state_dict'])
            except: self.renderer.module.load_state_dict(ckpt["audionerf_network_state_dict"])
            
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.renderer.parameters()), lr=float(self.kwargs_train['lr']),
                                          weight_decay=float(self.kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=float(self.kwargs_train['T_max']), eta_min=float(self.kwargs_train['eta_min']))
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']


    def save_checkpoint(self):
        """save model checkpoint

        Returns
        -------
        checkpoint name
        """
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))

        #if torch.cuda.device_count() > 1: state_dict = self.renderer.module.state_dict()
        #else: state_dict = self.renderer.state_dict()

        if dist.get_rank() == 0:  # メインプロセスのみ保存
            state_dict = self.renderer.module.state_dict() if hasattr(self.renderer, "module") else self.renderer.state_dict()
            torch.save({
                'current_iteration': self.current_iteration,
                'audionerf_network_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.cosine_scheduler.state_dict()
            }, ckptname)

        
        #torch.save({
        #    'current_iteration': self.current_iteration,
        #    'audionerf_network_state_dict': state_dict,
        #    'optimizer_state_dict': self.optimizer.state_dict(),
        #    'scheduler_state_dict': self.cosine_scheduler.state_dict()
        #}, ckptname)
        return ckptname


    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current Iteration:%d", self.current_iteration)
                
        while self.current_iteration <= self.total_iterations:
            # 各エポックの開始時に `set_epoch()` を呼ぶ
            self.train_sampler.set_epoch(self.current_iteration)

            num_epochs = self.total_iterations // len(self.train_iter)
            print(f"トータル回数: {self.total_iterations}, バッチ回数: {len(self.train_iter)}, エポック数: {num_epochs}")
            num_train_samples = len(self.train_iter.dataset)
            num_test_samples = len(self.test_iter.dataset)

            print(f"Train samples: {num_train_samples}, Test samples: {num_test_samples}")
            
            with tqdm(total=len(self.train_iter), desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                for train_batch in self.train_iter:
                    log_gpu_memory(f"Iter {self.current_iteration} - Before renderer")
                    torch.cuda.empty_cache()
                    if self.dataset_type == "RAF":
                        ori_sig, position_rx, position_tx, direction_tx = train_batch
                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                    else:
                        ori_sig, position_rx, position_tx = train_batch
                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                    
                    torch.cuda.empty_cache()
                    log_gpu_memory(f"Iter {self.current_iteration} - After renderer")
                    
                    pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                    ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                    torch.cuda.empty_cache()

                    spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, _, _ = self.criterion(pred_sig, ori_sig)

                    torch.cuda.empty_cache()
        
                    if torch.isnan(energy_loss).item():
                        print("Nan loss detected")
                        continue

                    total_loss = spec_loss + amplitude_loss + angle_loss + time_loss + energy_loss + multi_stft_loss
                    
                    torch.cuda.empty_cache()

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.cuda.empty_cache()

                    torch.nn.utils.clip_grad_norm_(self.renderer.parameters(), max_norm=1)
                    for param in self.renderer.parameters():
                        if param.grad is not None:
                            with torch.no_grad():
                                param.grad[param.grad != param.grad] = 0
                                param.grad[torch.isinf(param.grad)] = 0

                    self.optimizer.step()
                    self.cosine_scheduler.step()
                    self.current_iteration += 1

                    if self.current_iteration % 20 == 0:
                        self.writer.add_scalar(f'train_loss', total_loss.detach(), self.current_iteration)

                        for param_group in self.optimizer.param_groups:
                            current_lr = param_group['lr']
                            self.writer.add_scalar(f'learning rate', current_lr, self.current_iteration)

                    pbar.update(1)
                    pbar.set_description(f"{self.expname} Iteration {self.current_iteration}/{self.total_iterations}")
                    pbar.set_postfix_str('loss = {:.4f}, multi stft loss:{:.4f}, spec loss:{:.3f}, amp loss:{:.3f}, angle loss:{:.3f}, time loss:{:.3f}, energy loss:{:.3f}, lr = {:.6f}'.format( \
                        total_loss.item(), multi_stft_loss, spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, self.optimizer.param_groups[0]['lr']))

                    if self.current_iteration % self.save_freq == 0:
                        ckptname = self.save_checkpoint()
                        pbar.write('Saved checkpoints at {}'.format(ckptname))

                    if self.current_iteration % self.val_freq == 0:
                        self.logger.info("Start evaluation")
                        self.renderer.eval()

                        valid_losses = {'spec_loss': 0, 'fft_loss': 0, 'time_loss': 0, 'energy_loss': 0, 'multi_stft_loss': 0}
                        valid_metrics = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0, 'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}
                        valid_metrics_for_std = {'Angle': [], 'Amplitude': [], 'Envelope': [], 'T60': [], 'C50': [], 'EDT': [], 'multi_stft': []}

                        for check_idx, test_batch in enumerate(self.test_iter):
                            with torch.no_grad():
                                if self.dataset_type == "RAF":
                                    ori_sig, position_rx, position_tx, direction_tx = test_batch
                                    pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                                else:
                                    ori_sig, position_rx, position_tx = test_batch
                                    pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                                                                                
                                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                                losses, metrics, ori_time, pred_time = self.calculate_metrics(pred_sig, ori_sig, self.fs)

                            for key in valid_losses:
                                valid_losses[key] += losses[key].detach()

                            for key in valid_metrics:
                                valid_metrics[key] += metrics[key]
                            
                            for key in valid_metrics_for_std:
                                valid_metrics_for_std[key].append(metrics[key])

                            if check_idx < 15:                                
                                save_dir = os.path.join(self.logdir, self.expname, f'img_test/{str(self.current_iteration//1000).zfill(4)}_{str(check_idx).zfill(5)}.png')
                                plot_and_save_figure(pred_sig[0,:], ori_sig[0,:], pred_time[0,:], ori_time[0,:], position_rx[0,:], position_tx[0,:], mode_set='test', save_path=save_dir)
                            
                                save_dir = os.path.join(self.logdir, self.expname, f'img_test/energy_{str(self.current_iteration//1000).zfill(4)}_{str(check_idx).zfill(5)}.png')      
                                log_inference_figure(ori_time.detach().cpu().numpy()[0,:], pred_time.detach().cpu().numpy()[0,:], metrics=metrics, save_dir=save_dir)

                        num_batches = len(self.test_iter)
                        avg_losses = {key: valid_losses[key] / num_batches for key in valid_losses}
                        avg_metrics = {key: valid_metrics[key] / num_batches for key in valid_metrics}
                        std_metrics = {key: np.std(valid_metrics_for_std[key]) for key in valid_metrics_for_std}

                        self.log_tensorboard(losses=avg_losses, metrics=avg_metrics, metrics_std=std_metrics, cur_iter=self.current_iteration, mode_set="test")
                        self.logger.info("Evaluations. Current Iteration:%d", self.current_iteration)

                        self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
                        avg_metrics['Angle'], avg_metrics['Amplitude'], avg_metrics['Envelope'], avg_metrics['T60'], avg_metrics['C50'], avg_metrics['EDT'], avg_metrics['multi_stft']))

                        self.logger.info('STD Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
                        std_metrics['Angle'], std_metrics['Amplitude'], std_metrics['Envelope'], std_metrics['T60'], std_metrics['C50'], std_metrics['EDT'], std_metrics['multi_stft']))
                            
                        train_losses = {'spec_loss': 0, 'fft_loss': 0, 'time_loss': 0, 'energy_loss': 0, 'multi_stft_loss': 0}
                        train_metrics = {'Angle': 0, 'Amplitude': 0, 'Envelope': 0, 'T60': 0, 'C50': 0, 'EDT': 0, 'multi_stft': 0}
                        train_metrics_for_std = {'Angle': [], 'Amplitude': [], 'Envelope': [], 'T60': [], 'C50': [], 'EDT': [], 'multi_stft': []}

                        for check_idx, train_iter_batch in enumerate(self.train_iter_show):
                            with torch.no_grad():
                                if self.dataset_type == "RAF":
                                    ori_sig, position_rx, position_tx, direction_tx = train_iter_batch
                                    pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                                else:
                                    ori_sig, position_rx, position_tx = train_iter_batch
                                    pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                                                                                
                                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                                losses, metrics, ori_time, pred_time = self.calculate_metrics(pred_sig, ori_sig, self.fs)

                            for key in train_losses:
                                train_losses[key] += losses[key].detach()

                            for key in train_metrics:
                                train_metrics[key] += metrics[key]
                            
                            for key in train_metrics_for_std:
                                train_metrics_for_std[key].append(metrics[key])

                            if check_idx < 15:
                                save_dir = os.path.join(self.logdir, self.expname, f'img_train/{str(self.current_iteration//1000).zfill(4)}_{str(check_idx).zfill(5)}.png')
                                plot_and_save_figure(pred_sig[0,:], ori_sig[0,:], pred_time[0,:], ori_time[0,:], position_rx[0,:], position_tx[0,:], mode_set='train', save_path=save_dir)

                            if check_idx > 3000 or check_idx == len(self.train_iter_show) - 1:
                                num_batches = check_idx + 1
                                avg_losses = {key: train_losses[key] / num_batches for key in train_losses}
                                avg_metrics = {key: train_metrics[key] / num_batches for key in train_metrics}
                                std_metrics = {key: np.std(train_metrics_for_std[key]) for key in train_metrics_for_std}

                                self.log_tensorboard(losses=avg_losses, metrics=avg_metrics, metrics_std=std_metrics, cur_iter=self.current_iteration, mode_set="train")
                                
                                self.logger.info("Evaluations on training set")
                                self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
                                avg_metrics['Angle'], avg_metrics['Amplitude'], avg_metrics['Envelope'], avg_metrics['T60'], avg_metrics['C50'], avg_metrics['EDT'], avg_metrics['multi_stft']))

                                self.logger.info('Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
                                std_metrics['Angle'], std_metrics['Amplitude'], std_metrics['Envelope'], std_metrics['T60'], std_metrics['C50'], std_metrics['EDT'], std_metrics['multi_stft']))
                                
                                break

                        self.renderer.train()
    
    def calculate_metrics(self, pred_sig, ori_sig, fs):
        """ calculate the metrics and losses
        """
        # Loss calculation
        spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, ori_time, pred_time = self.criterion(pred_sig, ori_sig)

        # Metrics calculation
        angle_error, amp_error, env_error, t60_error, edt_error, C50_error, multi_stft, _, _ = metric_cal(
            ori_time.detach().cpu().numpy(), 
            pred_time.detach().cpu().numpy(), 
            fs=fs
        )

        losses = {
            'spec_loss': spec_loss,
            'fft_loss': amplitude_loss + angle_loss,
            'time_loss': time_loss,
            'energy_loss': energy_loss,
            'multi_stft_loss': multi_stft_loss
        }

        metrics = {
            'Angle': angle_error,
            'Amplitude': amp_error,
            'Envelope': env_error,
            'T60': t60_error,
            'C50': C50_error,
            'EDT': edt_error,
            'multi_stft': multi_stft
        }

        return losses, metrics, ori_time, pred_time

    def log_tensorboard(self, losses=None, metrics=None, metrics_std=None, cur_iter=None, mode_set="train"):
        for loss_name, value in losses.items():
            self.writer.add_scalar(f'{mode_set}_loss/{loss_name}', value, cur_iter)
    
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{mode_set}_metric/{metric_name}', value, cur_iter)
        
        for metric_name, value in metrics_std.items():
            self.writer.add_scalar(f'{mode_set}_metric_std/{metric_name}', value, cur_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=str, default='avr.yml', help='config file path')
    parser.add_argument('--dataset_dir', type=str, default='S1-M3969_npy')
    parser.add_argument('--batch_size', type=int, default=3)
    args = parser.parse_args()


    if args.mode == 'train': # specify the config yaml
        with open(args.config, 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)
    elif args.mode =='test': # specify the dict of the config yaml
        with open(os.path.join(args.config, 'avr_conf.yml'), 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

    ## backup config file
    logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
    os.makedirs(logdir, exist_ok=True)

    # Log the command to the file
    logfile_path = "command_log.txt"
    command = " ".join(sys.argv)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(logdir, logfile_path), "a") as logfile:
        logfile.write(f"[{current_time}] : {command}\n")

    # Construct the destination file path
    dest_file_path = os.path.join(logdir, 'avr_conf.yml')

    # create the img path
    img_train_dir = os.path.join(logdir, 'img_train')
    os.makedirs(img_train_dir, exist_ok=True)

    img_test_dir = os.path.join(logdir, 'img_test')
    os.makedirs(img_test_dir, exist_ok=True)

    # Check if the source and destination paths are the same
    if os.path.abspath(args.config) != os.path.abspath(dest_file_path):
        copyfile(args.config, dest_file_path)
    else:
        print("Source and destination are the same, skipping copy.")

    world_size = int(os.environ["WORLD_SIZE"])  # DDP環境変数から取得
    
    # batch size は 各GPUでのローカルバッチサイズにすべき
    per_gpu_batch_size = args.batch_size // world_size
    
    #worker = AVR_Runner(mode=args.mode, dataset_dir=args.dataset_dir, batchsize=args.batch_size, **kwargs)
    worker = AVR_Runner(mode=args.mode, dataset_dir=args.dataset_dir, batchsize=per_gpu_batch_size, **kwargs)
    worker.train()

    # DDPの終了処理
    dist.destroy_process_group()
