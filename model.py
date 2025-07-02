"""piowave network model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
import math

class AVRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        dir_encoding_sig = cfg["dir_encoding_sig"]
        tx_encoding_sig = cfg["tx_encoding_sig"]

        sigma_encoder_network = cfg["sigma_encoder_network"]
        sigma_decoder_network = cfg["sigma_decoder_network"]
        signal_network = cfg['signal_network']

        self.signal_output_dim = cfg['signal_output_dim']

        self._pos_encoding = tcnn.Encoding(3, pos_encoding_sigma, dtype=torch.float32)
        self._dir_encoding = tcnn.Encoding(3, dir_encoding_sig, dtype=torch.float32)
        self._tx_encoding = tcnn.Encoding(3, tx_encoding_sig, dtype=torch.float32)

        channel_cfg = cfg.get("channel_embed")
        if channel_cfg is not None and channel_cfg.get("is_embed", False):
            self.ch_num = channel_cfg["ch_num"]
            self.channel_emb_dim = channel_cfg.get("emb_dim", 128)
            self.channel_embedding = nn.Parameter(
                torch.randn(self.ch_num, self.channel_emb_dim) / math.sqrt(self.channel_emb_dim),
                requires_grad=True
            )

            self.encoder_sigma_channel_emb_dim = self.channel_emb_dim
            self.signal_channel_emb_dim = self.channel_emb_dim
        else:
            self.encoder_sigma_channel_emb_dim = 0
            self.signal_channel_emb_dim = 0


        network_in_dims = self._pos_encoding.n_output_dims + self.encoder_sigma_channel_emb_dim
        self._model_encoder_sigma = tcnn.Network(
            n_input_dims=network_in_dims,
            n_output_dims=128,
            network_config=sigma_encoder_network,
        )

        self._model_decoder_sigma = tcnn.Network(
            n_input_dims=self._model_encoder_sigma.n_output_dims,
            n_output_dims=1,
            network_config=sigma_decoder_network,
        )

        n_signal_input = (
            self._model_encoder_sigma.n_output_dims +
            self._dir_encoding.n_output_dims +
            self._tx_encoding.n_output_dims +
            self.signal_channel_emb_dim
        )
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=signal_network,
        )

    def forward(self, pts, view, tx, ch_idx=None):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_rays * n_samples, 3], position of voxels
        view: [batchsize, n_rays * n_samples, 3], view direction
        tx: [batchsize, n_rays * n_samples, 3], position of transmitter, implemented as orientation of the rx data
        ch_idx: [batchsize], optional. channel index (int64) for each IR

        Returns
        ----------
        attn: [batchsize, n_rays * n_samples, 1].
        signal: [batchsize, n_rays * n_samples, ir length].
        """
        bs = pts.size(0)
        n_ray_points = pts.size(1)

        pts = (pts.view(-1,3) + 1)/2
        view = (view.view(-1,3) + 1)/2
        tx = (tx.view(-1,3) + 1)/2

        pos_embedding = self._pos_encoding(pts)

        # === 追加: チャンネル埋め込みの展開と結合 ===
        if hasattr(self, "channel_embedding") and ch_idx is not None:
            # ch_idx: [B] → [B, N] → [B*N]
            ch_idx_flat = ch_idx.unsqueeze(1).expand(-1, n_ray_points).reshape(-1)  # [B*N]
            # チャンネルベクトルの参照（[C, D] の中から ch_idx_flat で選ぶ）
            ch_embed = self.channel_embedding[ch_idx_flat]  # [B*N, channel_emb_dim]
            # encoder sigma 入力に結合
            encoder_input = torch.cat([pos_embedding, ch_embed], dim=-1)  # [B*N, D+channel_emb_dim]
        else:
            encoder_input = pos_embedding
            ch_embed = None  # signal への入力構築のために保持

        sigma_feature = self._model_encoder_sigma(encoder_input)
        attn = self._model_decoder_sigma(F.relu(sigma_feature))

        view_embedding = self._dir_encoding(view)
        tx_embedding = self._tx_encoding(tx)

        signal_inputs = [F.relu(sigma_feature), view_embedding, tx_embedding]
        if ch_embed is not None:
            signal_inputs.append(ch_embed)  # === 追加: signal ネットにも channel embedding を結合
    
        feature_all = torch.cat(signal_inputs, dim=-1)  # [B*N, total_dim]
        signal = self._model_signal(feature_all)

        attn = abs(F.leaky_relu(attn)).view(bs, n_ray_points, 1)
        signal = signal.reshape(bs, n_ray_points,  self.signal_output_dim)
        return attn, signal
    

class AVRModel_complex(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.leaky_relu = cfg["leaky_relu"]
        pos_encoding_sigma = cfg["pos_encoding_sigma"]
        tx_pos_encoding_sigma = cfg["tx_pos_encoding_sigma"]

        pos_encoding_signal = cfg['pos_encoding_sig']
        tx_pos_encoding_signal = cfg['tx_pos_encoding_sig']

        dir_encoding_sig = cfg["dir_encoding_sig"]
        tx_dir_encoding_sig = cfg["tx_dir_encoding_sig"]

        sigma_encoder_network = cfg["sigma_encoder_network"]
        sigma_decoder_network = cfg["sigma_decoder_network"]
        signal_network = cfg['signal_network']

        self.signal_output_dim = cfg['signal_output_dim']

        self._pos_encoding = tcnn.Encoding(3, pos_encoding_sigma, dtype=torch.float32)
        self._pos_signal_encoding = tcnn.Encoding(3, pos_encoding_signal, dtype=torch.float32)
        self._tx_pos_encoding = tcnn.Encoding(3, tx_pos_encoding_sigma, dtype=torch.float32)
        self._tx_pos_signal_encoding = tcnn.Encoding(3, tx_pos_encoding_signal, dtype=torch.float32)

        self._dir_encoding = tcnn.Encoding(3, dir_encoding_sig, dtype=torch.float32)
        self._tx_dir_encoding = tcnn.Encoding(3, tx_dir_encoding_sig, dtype=torch.float32)

        network_in_dims = self._pos_encoding.n_output_dims + self._tx_pos_encoding.n_output_dims
        self._model_encoder_sigma = tcnn.Network(
            n_input_dims=network_in_dims,
            n_output_dims=256,
            network_config=sigma_encoder_network,
        )

        self._model_decoder_sigma = tcnn.Network(
            n_input_dims=self._model_encoder_sigma.n_output_dims,
            n_output_dims=1,
            network_config=sigma_decoder_network,
        )

        n_signal_input = self._model_encoder_sigma.n_output_dims + \
                self._dir_encoding.n_output_dims + \
                self._tx_dir_encoding.n_output_dims + \
                self._pos_signal_encoding.n_output_dims + \
                self._tx_pos_signal_encoding.n_output_dims
        
        self._model_signal = tcnn.Network(
            n_input_dims=n_signal_input,
            n_output_dims=self.signal_output_dim,
            network_config=signal_network,
        )

    def forward(self, pts, view, tx, tx_view):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_rays * n_samples, 3], position of voxels
        view: [batchsize, n_rays * n_samples, 3], view direction
        tx: [batchsize, n_rays * n_samples, 3], position of emitter
        tx_view: [batchsize, n_rays * n_samples, 3], emitter view direction

        Returns
        ----------
        attn: [batchsize, n_rays * n_samples, 1].
        signal: [batchsize, n_rays * n_samples, ir length].
        """

        bs = pts.size(0)
        n_ray_points = pts.size(1)

        pts = (pts.view(-1,3) + 1)/2
        view = (view.view(-1,3) + 1)/2
        tx = (tx.view(-1,3) + 1)/2
        tx_view = (tx_view.reshape(-1,3) + 1)/2

        pos_embedding = self._pos_encoding(pts)
        tx_pos_embedding = self._tx_pos_encoding(tx)

        sigma_feature = self._model_encoder_sigma(torch.cat([pos_embedding, tx_pos_embedding], -1))
        attn = self._model_decoder_sigma(F.relu(sigma_feature))

        view_embedding = self._dir_encoding(view)
        tx_view_embedding = self._tx_dir_encoding(tx_view)
        signal_embedding = self._pos_signal_encoding(pts)
        tx_signal_embedding = self._tx_pos_signal_encoding(tx)

        feature_all = torch.cat([F.relu(sigma_feature), view_embedding, tx_view_embedding, signal_embedding, tx_signal_embedding], -1)
        signal = self._model_signal(feature_all)

        attn = abs(F.leaky_relu(attn, negative_slope=self.leaky_relu)).view(bs, n_ray_points, 1)
        signal = (signal).reshape(bs, n_ray_points,  self.signal_output_dim)
        return attn, signal