"""piowave network model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
import math


class LayeredTCNNWithInjection(nn.Module):
    def __init__(self, n_input_dims, n_neurons, n_hidden_layers, n_output_dims,
                 ch_num, activation="ReLU", otype="FullyFusedMLP"):
        super().__init__()
        self.activation = getattr(F, activation.lower())
        self.hidden_layers = nn.ModuleList()
        self.layer_embeddings = nn.ParameterList()

        in_dim = n_input_dims
        for i in range(n_hidden_layers):
            layer = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=n_neurons,
                network_config={
                    "otype": otype,
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": n_neurons,
                    "n_hidden_layers": 0
                }
            )
            self.hidden_layers.append(layer)

            emb = nn.Parameter(
                torch.randn(ch_num, n_neurons) / math.sqrt(n_neurons),
                requires_grad=True
            )
            self.layer_embeddings.append(emb)

            in_dim = n_neurons

        # 出力層
        self.output_layer = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=n_output_dims,
            network_config={
                "otype": otype,
                "activation": "None",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": 0
            }
        )

    def forward(self, x, ch_id=None):
        for idx, layer in enumerate(self.hidden_layers):
            h = layer(x)
            if ch_id is not None:
                h = h + self.layer_embeddings[idx][ch_id]
            x = self.activation(h)
        return self.output_layer(x)

class AVRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._pos_encoding = tcnn.Encoding(3, cfg["pos_encoding_sigma"])
        self._dir_encoding = tcnn.Encoding(3, cfg["dir_encoding_sig"])
        self._tx_encoding = tcnn.Encoding(3, cfg["tx_encoding_sig"])
        self.signal_output_dim = cfg["signal_output_dim"]

        ch = cfg.get("channel_embed") or {}
        is_embed = ch.get("is_embed", False)
        conn_type = ch.get("connection_type", None)  # 'add' or 'concat'
        self.ch_num = ch.get("ch_num", 0)

        self.emb_dim_enc = ch.get("emb_dim_sigma_encoder", 0)
        self.emb_dim_dec = ch.get("emb_dim_sigma_decoder", 0)
        self.emb_dim_sig = ch.get("emb_dim_signal_network", 0)

        flag_enc = bool(ch.get("is_sigma_encoder", False))
        flag_dec = bool(ch.get("is_sigma_decoder", False))
        flag_sig = bool(ch.get("is_signal_network", False))

        self.enc_injection = is_embed and (conn_type == "add")    and flag_enc
        self.dec_injection = is_embed and (conn_type == "add")    and flag_dec
        self.sig_injection = is_embed and (conn_type == "add")    and flag_sig
        self.enc_concat    = is_embed and (conn_type == "concat") and flag_enc
        self.dec_concat    = is_embed and (conn_type == "concat") and flag_dec
        self.sig_concat    = is_embed and (conn_type == "concat") and flag_sig

        sigma_encoder_cfg = cfg["sigma_encoder_network"]
        sigma_decoder_cfg = cfg["sigma_decoder_network"]
        signal_network_cfg = cfg["signal_network"]
        
        # ===== Sigma Encoder =====
        if self.enc_injection:
            self._model_encoder_sigma = LayeredTCNNWithInjection(
                n_input_dims=self._pos_encoding.n_output_dims,
                n_neurons=sigma_encoder_cfg["n_neurons"],
                n_hidden_layers=sigma_encoder_cfg["n_hidden_layers"],
                n_output_dims=128,
                ch_num=self.ch_num,
                activation=sigma_encoder_cfg.get("activation", "ReLU"),
                otype=sigma_encoder_cfg.get("otype", "FullyFusedMLP")
            )
            self.encoder_mode = "injection"
        else:
            if self.enc_concat:
                self.encoder_channel_embedding = nn.Parameter(
                    torch.randn(self.ch_num, self.emb_dim_enc) / math.sqrt(self.emb_dim_enc),
                    requires_grad=True
                )
                enc_in = self._pos_encoding.n_output_dims + self.emb_dim_enc
            else:
                enc_in = self._pos_encoding.n_output_dims

            self._model_encoder_sigma = tcnn.Network(
                n_input_dims=enc_in,
                n_output_dims=128,
                network_config=sigma_encoder_cfg
            )
            self.encoder_mode = "concat" if self.enc_concat else "none"

        # ===== Sigma Decoder =====
        if self.dec_injection:
            self._model_decoder_sigma = LayeredTCNNWithInjection(
                n_input_dims=128,
                n_neurons=sigma_decoder_cfg["n_neurons"],
                n_hidden_layers=sigma_decoder_cfg["n_hidden_layers"],
                n_output_dims=1,
                ch_num=self.ch_num,
                activation=sigma_decoder_cfg.get("activation", "ReLU"),
                otype=sigma_decoder_cfg.get("otype", "FullyFusedMLP")
            )
            self.decoder_mode = "injection"
        else:
            if self.dec_concat:
                self.decoder_channel_embedding = nn.Parameter(
                    torch.randn(self.ch_num, self.emb_dim_dec) / math.sqrt(self.emb_dim_dec),
                    requires_grad=True
                )
                dec_in = 128 + self.emb_dim_dec
            else:
                dec_in = 128

            self._model_decoder_sigma = tcnn.Network(
                n_input_dims=dec_in,
                n_output_dims=1,
                network_config=sigma_decoder_cfg
            )
            self.decoder_mode = "concat" if self.dec_concat else "none"
        
        # ===== Signal Network =====
        base_sig_in = 128 + self._dir_encoding.n_output_dims + self._tx_encoding.n_output_dims
        if self.sig_injection:
            self._model_signal = LayeredTCNNWithInjection(
                n_input_dims=base_sig_in,
                n_neurons=signal_network_cfg["n_neurons"],
                n_hidden_layers=signal_network_cfg["n_hidden_layers"],
                n_output_dims=self.signal_output_dim,
                ch_num=self.ch_num,
                activation=signal_network_cfg.get("activation", "ReLU"),
                otype=signal_network_cfg.get("otype", "CutlassMLP")
            )
            self.signal_mode = "injection"
        else:
            if self.sig_concat:
                self.signal_channel_embedding = nn.Parameter(
                    torch.randn(self.ch_num, self.emb_dim_sig) / math.sqrt(self.emb_dim_sig),
                    requires_grad=True
                )
                sig_in = base_sig_in + self.emb_dim_sig
            else:
                sig_in = base_sig_in

            self._model_signal = tcnn.Network(
                n_input_dims=sig_in,
                n_output_dims=self.signal_output_dim,
                network_config=signal_network_cfg
            )
            self.signal_mode = "concat" if self.sig_concat else "none"

    def forward(self, pts, view, tx, ch_idx=None):
        bs = pts.size(0)
        n_ray_points = pts.size(1)

        pts = (pts.view(-1, 3) + 1) / 2
        view = (view.view(-1, 3) + 1) / 2
        tx = (tx.view(-1, 3) + 1) / 2

        pos_enc = self._pos_encoding(pts)

        ch_idx_expanded = None
        if ch_idx is not None:
            ch_idx_expanded = ch_idx.unsqueeze(1).expand(-1, n_ray_points).reshape(-1)

        # === Sigma Encoder ===
        if self.encoder_mode == "injection":
            sigma_feat = self._model_encoder_sigma(pos_enc, ch_idx_expanded)
        else:
            if self.encoder_mode == "concat" and ch_idx_expanded is not None:
                enc_emb = self.encoder_channel_embedding[ch_idx_expanded]
                enc_in = torch.cat([pos_enc, enc_emb], dim=-1)
            else:
                enc_in = pos_enc
            sigma_feat = self._model_encoder_sigma(enc_in)

        # ----- Sigma Decoder -----
        dec_in = F.relu(sigma_feat)
        if self.decoder_mode == "injection":
            attn = self._model_decoder_sigma(dec_in, ch_idx_expanded)
        else:
            if self.decoder_mode == "concat" and ch_idx_expanded is not None:
                dec_emb = self.decoder_channel_embedding[ch_idx_expanded]
                dec_in = torch.cat([dec_in, dec_emb], dim=-1)
            attn = self._model_decoder_sigma(dec_in)

        # === Signal network ===
        dir_enc = self._dir_encoding(view)
        tx_enc = self._tx_encoding(tx)
        signal_base = torch.cat([sigma_feat, dir_enc, tx_enc], dim=-1)

        if self.signal_mode == "injection":
            signal_output = self._model_signal(signal_base, ch_idx_expanded)
        else:
            if self.signal_mode == "concat" and ch_idx_expanded is not None:
                sig_emb = self.signal_channel_embedding[ch_idx_expanded]
                signal_in = torch.cat([signal_base, sig_emb], dim=-1)
            else:
                signal_in = signal_base
            signal_output = self._model_signal(signal_in)

        attn = abs(F.leaky_relu(attn)).view(bs, n_ray_points, 1)
        signal_output = signal_output.view(bs, n_ray_points, self.signal_output_dim)
        return attn, signal_output


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