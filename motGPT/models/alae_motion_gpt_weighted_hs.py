"""Text-to-motion via LLM *layer-routed weighted* hidden states + frozen ALAE decoder.

Router edition of :class:`ALAEMotionGPT` (2026-07-09 rework). The ONLY change
vs the parent is WHERE the per-slot hidden state comes from: instead of the
LLM's last transformer layer, each latent slot k gets a learned soft routing
distribution over all L layers and combines that step's per-layer hidden
states with it. Everything downstream is inherited unchanged -- the Linear
projector -> (mu, logvar), the low-rank factor head (latent_low_rank > 0,
amplitudes read from the slot-pooled routed hidden), the NLL/chamfer anchors,
the GNU M2T branch and its joint loss -- so any parent config runs by just
switching ``model.target`` and adding the router params. This is deliberate:
against an NLL/FAC baseline the router is a one-change A/B (same head, same
losses, only the hidden source differs).

Two branches (both reuse the same first LLM forward over the prompt):

  Branch A (routing weights):
      GNU/T2M prompt -> LLM -> last-layer hidden (B, T, H)
                 -> masked mean over non-pad tokens -> (B, H)
                 -> Linear H->D -> memory (B, 1, D)
      queries = stage2_latent_queries (K, D) [imported from Stage-1 ALAE ckpt]
      cross-attention(queries, memory) -> (B, K, D)
      shared MLP router -> (B, K, L) logits
      train: Gumbel-Softmax over L (tau annealed) -> weights (B, K, L)
      eval:  deterministic softmax(logits / tau_infer), NO Gumbel noise

  Branch B (hidden states):
      AR-roll K steps (feed back last-layer hidden through feedback_norm,
      exactly the parent rollout), collect ALL L block outputs per step
      -> H_all (B, K, L, H)

  Combine:
      weighted = einsum('bkl,bklh->bkh', weights, H_all)   # (B, K, H)
      -> returned by `_encode_text_ar` -> parent heads as usual.

L = ``language_model.config.num_hidden_layers`` (GPT-2: 12), never hardcoded.

Eval-time tau: detached inference (loading a ckpt without a fit loop) has
``current_epoch == 0``, so the annealed schedule would silently give
``tau_start`` (near-uniform routing). Eval therefore always uses the
converged ``gumbel_tau_end`` unless ``router_eval_tau`` overrides it; the
annealed value is used for the TRAIN-time Gumbel draw only.

Incompatibilities (guarded in __init__): ``ar_latent_sampling`` feeds LATENT
values back through the chain, which bypasses the routed rollout entirely.

History: the 2026-06 version of this class carried its own MLP head
(``weighted_projector``) and deleted the parent's projector, which made it
incompatible with the factor head (AttributeError) and dropped the GNU M2T
loss in its ``allsplit_step``. June checkpoints load only with the June code
(git history); this rework targets the NLL/FAC recipe.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from motGPT.models.alae_motion_gpt import ALAEMotionGPT
from adaptive_length_auto_encoder.model import (
    CrossAttentionBlock,
    latent_query_orthogonality_loss,
)


class ALAEMotionGPTWeightedHS(ALAEMotionGPT):
    """LLM + ALAE-decoder model with layer-routed weighted hidden states."""

    def __init__(
        self,
        *args,
        num_cross_attn_layers: int = 2,
        router_hidden: int = 512,
        gumbel_tau_start: float = 5.0,
        gumbel_tau_end: float = 0.5,
        gumbel_anneal_epochs: int = 50,
        lambda_ortho: float = 1.0,
        freeze_latent_queries: bool = True,
        router_eval_tau: Optional[float] = None,
        **kwargs,
    ):
        # Cross-attention hyperparameters mirror the Stage-1 ALAE encoder; read
        # them from the `alae` sub-config so the block matches the ckpt's style.
        alae_cfg = dict(kwargs.get('alae') or {})

        super().__init__(*args, **kwargs)

        if self.ar_latent_sampling:
            raise ValueError(
                'ALAEMotionGPTWeightedHS is incompatible with ar_latent_sampling: '
                'the chain feeds back latent VALUES, which bypasses the routed '
                'hidden-state rollout.')

        self.num_cross_attn_layers = int(num_cross_attn_layers)
        self.router_hidden = int(router_hidden)
        self.gumbel_tau_start = float(gumbel_tau_start)
        self.gumbel_tau_end = float(gumbel_tau_end)
        self.gumbel_anneal_epochs = int(gumbel_anneal_epochs)
        self.lambda_ortho = float(lambda_ortho)
        self.freeze_latent_queries = bool(freeze_latent_queries)
        self.router_eval_tau = (
            None if router_eval_tau is None else float(router_eval_tau))

        H = self.language_model.config.hidden_size
        D = self.alae_wrapper.latent_dim
        self.n_layers = int(self.language_model.config.num_hidden_layers)

        nhead = int(alae_cfg.get('nhead', 8))
        dim_feedforward = int(alae_cfg.get('dim_feedforward', 1024))
        dropout = float(alae_cfg.get('dropout', 0.1))
        activation = str(alae_cfg.get('activation', 'gelu'))

        # NB: unlike the June version, the parent's projector / projector_norm
        # (and the factor heads when latent_low_rank > 0) are KEPT -- they ARE
        # the head; the router only changes the hidden fed into them.

        # ----- Branch A: averaged text hidden -> cross-attn memory -----
        self.text_mem_proj = nn.Linear(H, D)

        # Latent queries imported from the Stage-1 ALAE ckpt (already loaded by
        # ALAEWrapper inside the parent __init__). Clone so we can train/freeze
        # them independently of the (frozen) encoder-side queries.
        src_queries = self.alae_wrapper.alae.latent_queries.detach().clone()
        self.stage2_latent_queries = nn.Parameter(src_queries)
        assert self.stage2_latent_queries.shape == self.alae_wrapper.alae.latent_queries.shape, (
            f'latent_queries shape mismatch: {self.stage2_latent_queries.shape} vs '
            f'{self.alae_wrapper.alae.latent_queries.shape}'
        )
        assert torch.equal(
            self.stage2_latent_queries.data, self.alae_wrapper.alae.latent_queries.data
        ), 'stage2_latent_queries must be imported from the ALAE ckpt, not re-initialised'
        if self.freeze_latent_queries:
            self.stage2_latent_queries.requires_grad_(False)

        self.cross_attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=D,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.num_cross_attn_layers)
            ]
        )
        self.cross_attn_norm = nn.LayerNorm(D)

        # ----- Shared MLP router: D -> L (applied per latent slot) -----
        self.layer_router = nn.Sequential(
            nn.Linear(D, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.n_layers),
        )

        # Filled by _encode_text_ar, consumed by forward_motion/allsplit_step.
        self._last_ortho = None
        self._last_router_stats = None
        self._last_router_weights = None

    # ------------------------------------------------------------------
    # Gumbel temperature schedule
    # ------------------------------------------------------------------
    def _gumbel_tau(self) -> float:
        """TRAIN-time annealed tau (linear tau_start -> tau_end by epoch)."""
        try:
            epoch = int(self.current_epoch)
        except (RuntimeError, AttributeError):
            epoch = 0
        if self.gumbel_anneal_epochs <= 0:
            return self.gumbel_tau_end
        frac = min(max(epoch / float(self.gumbel_anneal_epochs), 0.0), 1.0)
        return self.gumbel_tau_start + (self.gumbel_tau_end - self.gumbel_tau_start) * frac

    def _router_infer_tau(self) -> float:
        """EVAL-time tau: the converged tau_end (or the explicit override).

        Never the annealed value: detached inference has current_epoch == 0,
        which would silently give tau_start (near-uniform routing).
        """
        if self.router_eval_tau is not None:
            return self.router_eval_tau
        return self.gumbel_tau_end

    # ------------------------------------------------------------------
    # AR rollout: parent loop + all-layer collection + router combination
    # ------------------------------------------------------------------
    def _encode_text_ar(self, texts: List[str]) -> torch.Tensor:
        """Roll K steps and return the ROUTED per-slot hidden states.

        Overrides the parent (which returns the last layer only); the AR loop,
        prompt construction (GNU-aware via ``_build_t2m_inputs``) and the
        feedback path (last-layer hidden through ``feedback_norm``) are
        byte-compatible with the parent rollout.

        Returns
        -------
        weighted : (B, K, hidden_size) router-weighted hidden states.
        """
        K = int(self.hparams.k_motion_tokens)
        device = self.device

        prompts = self._build_t2m_inputs(texts)
        enc = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.lm.max_length,
        ).to(device)
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']

        use_cache = bool(self.hparams.use_kv_cache)

        # 1) First forward over the full prompt (left-padded; <MOT> is last col).
        out = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            return_dict=True,
        )

        # Branch A memory: masked mean of the last-layer hidden over non-pad tokens.
        last_layer = out.hidden_states[-1]  # (B, T, H)
        m = attention_mask.unsqueeze(-1).to(last_layer.dtype)  # (B, T, 1)
        avg_text_hidden = (last_layer * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B, H)

        # Branch B step 0: all L block outputs at the last column.
        def _stack_all_layers(hidden_states) -> torch.Tensor:
            # hidden_states: tuple of (L+1) tensors (B, seq, H); take [1:] (the L
            # block outputs), keep last column, stack on a new layer axis.
            cols = [h[:, -1:, :] for h in hidden_states[1:]]  # L x (B,1,H)
            return torch.stack(cols, dim=2)  # (B, 1, L, H)

        all_steps = [_stack_all_layers(out.hidden_states)]      # list of (B,1,L,H)
        last_hiddens = [out.hidden_states[-1][:, -1:, :]]        # list of (B,1,H)
        past = out.past_key_values if use_cache else None
        cur_attn = attention_mask
        prompt_embeds = None

        for _ in range(K - 1):
            next_input = self.feedback_norm(last_hiddens[-1])  # (B, 1, H)
            if use_cache:
                cur_attn = torch.cat(
                    [cur_attn, torch.ones((cur_attn.shape[0], 1), dtype=cur_attn.dtype, device=device)],
                    dim=1,
                )
                step = self.language_model(
                    inputs_embeds=next_input,
                    attention_mask=cur_attn,
                    past_key_values=past,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
                past = step.past_key_values
            else:
                if prompt_embeds is None:
                    prompt_embeds = self.language_model.get_input_embeddings()(input_ids)
                cat_h = torch.cat(last_hiddens, dim=1)  # (B, t, H)
                full_embeds = torch.cat([prompt_embeds, self.feedback_norm(cat_h)], dim=1)
                step_attn = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], cat_h.shape[1]), dtype=attention_mask.dtype, device=device),
                    ],
                    dim=1,
                )
                step = self.language_model(
                    inputs_embeds=full_embeds,
                    attention_mask=step_attn,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            all_steps.append(_stack_all_layers(step.hidden_states))
            last_hiddens.append(step.hidden_states[-1][:, -1:, :])

        H_all = torch.cat(all_steps, dim=1)  # (B, K, L, H)
        B = H_all.shape[0]

        # 2) Branch A: cross-attention queries (from ckpt) over the averaged memory.
        memory = self.text_mem_proj(avg_text_hidden).unsqueeze(1)  # (B, 1, D)
        latents = self.stage2_latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        for block in self.cross_attn_blocks:
            latents = block(latents, memory)
        latents = self.cross_attn_norm(latents)  # (B, K, D)

        logits = self.layer_router(latents)  # (B, K, L)
        # Compute routing in fp32 for stability under bf16 autocast, then cast
        # back to the hidden-state dtype for the weighted sum.
        if self.training:
            # Train: stochastic Gumbel-Softmax (anneal tau over epochs).
            tau = self._gumbel_tau()
            weights = F.gumbel_softmax(
                logits.float(), tau=tau, hard=False, dim=-1
            ).to(H_all.dtype)  # (B, K, L)
        else:
            # Eval: deterministic tempered softmax (NO Gumbel sampling). This
            # keeps the non-mm metrics (FID / R-precision / MMDist)
            # reproducible across replications instead of jittering with fresh
            # Gumbel noise every forward. mm-mode diversity is injected by the
            # latent-noise path in val_t2m_forward, not by the router.
            tau = self._router_infer_tau()
            weights = F.softmax(logits.float() / tau, dim=-1).to(H_all.dtype)

        weighted = torch.einsum('bkl,bklh->bkh', weights, H_all)  # (B, K, H)

        if self.stage2_latent_queries.requires_grad:
            self._last_ortho = latent_query_orthogonality_loss(self.stage2_latent_queries)
        else:
            self._last_ortho = None
        with torch.no_grad():
            w = weights.float().clamp_min(1e-9)
            entropy = -(w * w.log()).sum(dim=-1).mean()  # mean over B, K
            self._last_router_stats = {
                'entropy': entropy,
                'tau': entropy.new_tensor(float(tau)),
            }
        # Detached copy for post-hoc router-decision analysis hooks.
        self._last_router_weights = weights.detach()

        return weighted

    # ------------------------------------------------------------------
    # Loss: parent total (incl. GNU M2T fold-in via allsplit_step) + ortho
    # ------------------------------------------------------------------
    def forward_motion(self, batch):
        out = super().forward_motion(batch)
        ortho = self._last_ortho
        if ortho is not None and self.lambda_ortho != 0.0:
            out['loss'] = out['loss'] + self.lambda_ortho * ortho
            out['loss_ortho'] = ortho
        else:
            out['loss_ortho'] = out['loss'].new_zeros(())
        return out

    # ------------------------------------------------------------------
    # Lightning step: defer to the parent (keeps the GNU M2T joint loss and
    # the full log table), then append the router diagnostics.
    # ------------------------------------------------------------------
    def allsplit_step(self, split: str, batch, batch_idx):
        result = super().allsplit_step(split, batch, batch_idx)
        if split == 'train':
            batch_size = len(batch['text'])
            extras = {}
            if torch.is_tensor(self._last_ortho):
                extras['train/loss_ortho'] = self._last_ortho.detach()
            stats = self._last_router_stats
            if stats is not None:
                extras['train/router_entropy'] = stats['entropy']
                extras['train/router_tau'] = stats['tau']
            for name, val in extras.items():
                self.log(
                    name, val,
                    on_step=False, on_epoch=True, prog_bar=False,
                    logger=True, sync_dist=True, batch_size=batch_size,
                )
        return result
