import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from motGPT.archs.tools.resnet import Resnet1D


def build_sine_position_encoding(length: int, dim: int, device, dtype=torch.float32):
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f'Unsupported activation: {activation}')

    def forward(self, target, memory):
        x = self.norm1(target)
        x_sa, _ = self.self_attn(x, x, x, need_weights=False)
        target = target + self.drop1(x_sa)

        x = self.norm2(target)
        x_ca, _ = self.cross_attn(x, memory, memory, need_weights=False)
        target = target + self.drop2(x_ca)

        x = self.norm3(target)
        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        target = target + self.drop3(x_ff)
        return target


class MotionEncoderBackbone(nn.Module):
    def __init__(
        self,
        input_dim=263,
        hidden_dim=512,
        latent_dim=512,
        depth=3,
        dilation_growth_rate=3,
        activation='gelu',
        norm=None,
        num_res_blocks=2,
    ):
        super().__init__()
        blocks = [
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU() if activation == 'gelu' else nn.ReLU() if activation == 'relu' else nn.SiLU(),
        ]
        for _ in range(num_res_blocks):
            blocks.append(
                Resnet1D(
                    hidden_dim,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=False,
                    activation=activation,
                    norm=norm,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.output_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        hidden = self.backbone(x)
        return self.output_proj(hidden)


class MotionDecoderRefiner(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        hidden_dim=512,
        output_dim=263,
        depth=3,
        dilation_growth_rate=3,
        activation='gelu',
        norm=None,
        num_res_blocks=2,
    ):
        super().__init__()
        blocks = [
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU() if activation == 'gelu' else nn.ReLU() if activation == 'relu' else nn.SiLU(),
        ]
        for _ in range(num_res_blocks):
            blocks.append(
                Resnet1D(
                    hidden_dim,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                )
            )
        blocks.extend(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU() if activation == 'gelu' else nn.ReLU() if activation == 'relu' else nn.SiLU(),
                nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def latent_query_orthogonality_loss(queries: torch.Tensor) -> torch.Tensor:
    """Encourage the K latent-query vectors to be mutually orthogonal.

    queries: (K, D) learnable query matrix.
    Returns the mean squared off-diagonal cosine similarity (scalar).
    Adapted from JYe16/GeoMotionGPT motGPT/losses/orthogonal.py.
    """
    n = queries.shape[0]
    if n < 2:
        return queries.new_zeros(())
    # fp32 + NaN safety: normalize each query to unit length. Under bf16/fp16
    # autocast the normalize can produce NaNs, so we guard against it.
    q = queries.float()
    norms = q.norm(dim=-1)
    if (norms < 1e-8).any():
        return queries.new_zeros(())
    q = F.normalize(q, p=2, dim=-1)
    gram = q @ q.t()                       # (K, K) cosine similarities
    eye = torch.eye(n, device=q.device, dtype=q.dtype)
    diff = gram - eye                      # diagonal -> 0, off-diagonal = cos
    loss = diff.pow(2).sum() / (n * (n - 1))
    if torch.isnan(loss):
        return queries.new_zeros(())
    return loss.to(queries.dtype)


class AdaptiveLengthAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim=263,
        k=8,
        latent_dim=512,
        hidden_dim=512,
        depth=3,
        dilation_growth_rate=3,
        activation='gelu',
        norm=None,
        num_res_blocks=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        max_decode_len=256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.latent_dim = latent_dim
        self.max_decode_len = max_decode_len

        self.encoder_backbone = MotionEncoderBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            num_res_blocks=num_res_blocks,
        )
        self.latent_queries = nn.Parameter(torch.randn(k, latent_dim) * 0.02)
        self.encoder_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=latent_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(latent_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=latent_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder_refiner = MotionDecoderRefiner(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            num_res_blocks=num_res_blocks,
        )

        self.register_buffer(
            'decoder_query_pe',
            build_sine_position_encoding(max_decode_len, latent_dim, device=torch.device('cpu')),
            persistent=False,
        )

    def ensure_decode_capacity(self, target_len):
        target_len = int(target_len)
        if target_len <= self.max_decode_len:
            return
        self.max_decode_len = target_len
        self.register_buffer(
            'decoder_query_pe',
            build_sine_position_encoding(target_len, self.latent_dim, device=torch.device('cpu')),
            persistent=False,
        )

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def encode(self, x):
        x_in = self.preprocess(x)
        memory = self.encoder_backbone(x_in).permute(0, 2, 1)
        batch_size = memory.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        latents = queries
        for block in self.encoder_blocks:
            latents = block(latents, memory)
        return self.encoder_norm(latents), memory

    def decode(self, latents, target_len):
        self.ensure_decode_capacity(target_len)
        # Use relative-phase positional encoding: regardless of target_len, the
        # query positions are evenly spread across the full PE table so that the
        # encoding represents the frame's relative phase within the clip rather
        # than its absolute frame index. This is critical for variable-length
        # training where the same absolute index means different things in clips
        # of different lengths.
        if target_len > 1:
            idx = torch.linspace(
                0,
                self.max_decode_len - 1,
                target_len,
                device=latents.device,
            ).round().long()
        else:
            idx = torch.zeros(target_len, device=latents.device, dtype=torch.long)
        query_pe = self.decoder_query_pe.to(device=latents.device, dtype=latents.dtype)[idx]
        queries = query_pe.unsqueeze(0).expand(latents.shape[0], -1, -1)
        decoded = queries
        for block in self.decoder_blocks:
            decoded = block(decoded, latents)
        decoded = self.decoder_norm(decoded)
        decoded = decoded.permute(0, 2, 1)
        x_hat = self.decoder_refiner(decoded)
        return self.postprocess(x_hat)

    def forward(self, x, target_len=None):
        if target_len is None:
            target_len = x.shape[1]
        latents, memory = self.encode(x)
        x_hat = self.decode(latents, target_len=target_len)
        if x_hat.shape[1] != target_len:
            raise RuntimeError(f'Decoder returned length {x_hat.shape[1]}, expected {target_len}')
        return x_hat, latents, memory

    def compute_loss(
        self,
        x,
        x_hat,
        latents,
        mask=None,
        lengths=None,
        t2m_move_enc=None,
        t2m_motion_enc=None,
        percept_fn=None,
        lambda_rec=1.0,
        lambda_ric=0.5,
        lambda_percept=0.5,
        lambda_latent_l2=1e-4,
        lambda_ortho=0.0,
        ric_slice=slice(4, 67),
        use_percept_cosine=False,
        lambda_percept_cosine=1.0,
    ):
        """Reconstruction loss with optional T2M perceptual term.

        Components (all averaged only over valid frames if ``mask`` is given):
        - ``loss_rec``    : SmoothL1 over the full 263-d feature.
        - ``loss_ric``    : SmoothL1 over ric_data slice [4:67] (explicit joint
                            positions; matches SnapMoGen's ``loss_explicit``).
        - ``loss_percept``: MSE between frozen perceptual embeddings of
                            ``x_hat`` and ``x``. The embedding comes from
                            ``percept_fn(motion, lengths)`` when provided
                            (e.g. the SnapMoGen evaluator fid_emb), else from
                            the HumanML3D T2M encoder pair when both are given.
        - ``loss_percept_cos``: ``mean(1 - cos(feat_fake, feat_real))`` over the
                            same T2M embeddings, added only when
                            ``use_percept_cosine`` is set. The MSE matches the
                            magnitude of the embeddings, the cosine term matches
                            their direction.
        - ``loss_latent_l2``: tiny L2 regulariser on the latent tokens.
        """
        if mask is None:
            loss_rec = F.smooth_l1_loss(x_hat, x)
            loss_ric = F.smooth_l1_loss(x_hat[..., ric_slice], x[..., ric_slice])
        else:
            m = mask.to(dtype=x_hat.dtype).unsqueeze(-1)
            denom_frames = m.sum().clamp_min(1.0)

            err_full = F.smooth_l1_loss(x_hat, x, reduction='none') * m
            loss_rec = err_full.sum() / (denom_frames * x.shape[-1])

            err_ric = (
                F.smooth_l1_loss(x_hat[..., ric_slice], x[..., ric_slice], reduction='none')
                * m
            )
            loss_ric = err_ric.sum() / (denom_frames * (ric_slice.stop - ric_slice.start))

        loss_latent_l2 = latents.pow(2).mean()

        # Orthogonality regulariser on the learnable latent-query vectors.
        # Computed in fp32 inside the helper for autocast stability.
        loss_ortho = latent_query_orthogonality_loss(self.latent_queries)

        total = (
            lambda_rec * loss_rec
            + lambda_ric * loss_ric
            + lambda_latent_l2 * loss_latent_l2
            + lambda_ortho * loss_ortho
        )

        loss_percept = x_hat.new_zeros(())
        loss_percept_cos = x_hat.new_zeros(())
        # Perceptual embeddings (evaluator fid_emb): kept so callers can build a
        # distribution-matching (MMD) loss without a second evaluator forward.
        # ``feat_fake`` carries grad to x_hat/decoder; ``feat_real`` is detached
        # (real-motion target). None when no perceptual encoder is active.
        feat_fake = None
        feat_real = None
        has_t2m_pair = t2m_move_enc is not None and t2m_motion_enc is not None
        if lengths is not None and (percept_fn is not None or has_t2m_pair):
            # Run frozen encoders in fp32 even under fp16/bf16 autocast.
            with torch.amp.autocast(device_type=x_hat.device.type, enabled=False):
                if percept_fn is not None:
                    feat_real = percept_fn(x.float(), lengths)
                    feat_fake = percept_fn(x_hat.float(), lengths)
                else:
                    # Lazy import to avoid circular dep with utils helpers.
                    from utils.load_t2m_encoders import compute_t2m_perceptual_features

                    feat_real = compute_t2m_perceptual_features(
                        x.float(), lengths, t2m_move_enc, t2m_motion_enc
                    )
                    feat_fake = compute_t2m_perceptual_features(
                        x_hat.float(), lengths, t2m_move_enc, t2m_motion_enc
                    )
                loss_percept = F.mse_loss(feat_fake, feat_real)
                if use_percept_cosine:
                    # 1 - cos so that minimising the loss maximises directional
                    # alignment of the two embeddings.
                    loss_percept_cos = (
                        1.0 - F.cosine_similarity(feat_fake, feat_real, dim=-1)
                    ).mean()
            total = total + lambda_percept * loss_percept
            if use_percept_cosine:
                total = total + lambda_percept_cosine * loss_percept_cos

        return {
            'loss': total,
            'loss_rec': loss_rec,
            'loss_ric': loss_ric,
            'loss_percept': loss_percept,
            'loss_percept_cos': loss_percept_cos,
            'loss_latent_l2': loss_latent_l2,
            'loss_ortho': loss_ortho,
        }