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

        if self.dataset_type == 'MeshRIR' or self.dataset_type == 'Simu' or self.dataset_type == 'Real_env':
            audionerf = AVRModel(kwargs_network) # network
        elif self.dataset_type == 'RAF':
            audionerf = AVRModel_complex(kwargs_network) # network

        self.renderer = AVRRender(networks_fn=audionerf, **kwargs_render) # renderer

        # multi gpu
        if torch.cuda.device_count() > 1: self.renderer = nn.DataParallel(self.renderer)
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
        self.current_iteration = 0
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

        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.train_iter_show = DataLoader(self.train_set_show, batch_size=self.batch_size, shuffle=False, num_workers=4)

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

        if torch.cuda.device_count() > 1: state_dict = self.renderer.module.state_dict()
        else: state_dict = self.renderer.state_dict()
        
        torch.save({
            'current_iteration': self.current_iteration,
            'audionerf_network_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname


    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current Iteration:%d", self.current_iteration)
                
        while self.current_iteration <= self.total_iterations:
            with tqdm(total=len(self.train_iter), desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                for train_batch in self.train_iter:
                    if self.dataset_type == "RAF":
                        ori_sig, position_rx, position_tx, direction_tx, ch_idx = train_batch
                        if ch_idx[0] is None:
                            pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                        else:
                            pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda(), ch_idx=ch_idx.cuda())
                    else:
                        ori_sig, position_rx, position_tx, ch_idx = train_batch
                        if ch_idx[0] is None:
                            pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                        else:
                            pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), ch_idx=ch_idx.cuda())
                    
                    pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                    ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                    spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, _, _ = self.criterion(pred_sig, ori_sig)
        
                    if torch.isnan(energy_loss).item():
                        print("Nan loss detected")
                        continue

                    total_loss = spec_loss + amplitude_loss + angle_loss + time_loss + energy_loss + multi_stft_loss
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()

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

                        # 評価結果保存用（for npz）
                        ori_sig_list = []
                        pred_sig_list = []
                        position_rx_list = []
                        position_tx_list = []
                        ch_idx_list = []

                        for check_idx, test_batch in enumerate(self.test_iter):
                            with torch.no_grad():
                                if self.dataset_type == "RAF":
                                    ori_sig, position_rx, position_tx, direction_tx, ch_idx = test_batch
                                    if ch_idx[0] is None:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                                    else:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda(), ch_idx=ch_idx.cuda())
                                else:
                                    ori_sig, position_rx, position_tx, ch_idx = test_batch
                                    if ch_idx[0] is None:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                                    else:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), ch_idx=ch_idx.cuda())
                                                                                
                                pred_sig = pred_sig[...,0] + 1j * pred_sig[...,1]
                                ori_sig = (ori_sig.cuda()).to(pred_sig.dtype)

                                # 蓄積（CPUに転送してnumpy化）
                                ori_sig_list.append(ori_sig.detach().cpu().numpy())
                                pred_sig_list.append(pred_sig.detach().cpu().numpy())
                                position_rx_list.append(position_rx.detach().cpu().numpy())
                                position_tx_list.append(position_tx.detach().cpu().numpy())
                                if ch_idx[0] is not None:
                                    ch_idx_list.append(ch_idx.detach().cpu().numpy())

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

                        npz_dir = os.path.join(self.logdir, self.expname, "val_result")
                        os.makedirs(npz_dir, exist_ok=True)
                        save_path = os.path.join(npz_dir, f"val_iter{self.current_iteration:06d}.npz")

                        # 結合して保存
                        if len(ch_idx_list) == 0:
                            np.savez_compressed(
                                save_path,
                                ori_sig=np.concatenate(ori_sig_list, axis=0),
                                pred_sig=np.concatenate(pred_sig_list, axis=0),
                                position_rx=np.concatenate(position_rx_list, axis=0),
                                position_tx=np.concatenate(position_tx_list, axis=0),
                                fs=self.fs
                            )
                        else:
                            np.savez_compressed(
                                save_path,
                                ori_sig=np.concatenate(ori_sig_list, axis=0),
                                pred_sig=np.concatenate(pred_sig_list, axis=0),
                                position_rx=np.concatenate(position_rx_list, axis=0),
                                position_tx=np.concatenate(position_tx_list, axis=0),
                                dh_idx=np.concatenate(ch_idx_list, axis=0),
                                fs=self.fs
                            )
                        self.logger.info(f"Saved val npz to {save_path}")
                        
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
                                    ori_sig, position_rx, position_tx, direction_tx, ch_idx = train_iter_batch
                                    if ch_idx[0] is None:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda())
                                    else:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), direction_tx.cuda(), ch_idx=ch_idx.cuda())
                                else:
                                    ori_sig, position_rx, position_tx, ch_idx = train_iter_batch
                                    if ch_idx[0] is None:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda())
                                    else:
                                        pred_sig = self.renderer(position_rx.cuda(), position_tx.cuda(), ch_idx=ch_idx.cuda())
                                                                                
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

                                self.logger.info('STD Angle:{:.3f}, Amplitude:{:.4f}, Envelope:{:.4f}, T60:{:.5f}, C50:{:.5f}, EDT:{:.5f}, multi_stft:{:.4f}'.format( \
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
    parser.add_argument('--batch_size', type=int, default=4)
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


    worker = AVR_Runner(mode=args.mode, dataset_dir=args.dataset_dir, batchsize=args.batch_size, **kwargs)
    worker.train()
