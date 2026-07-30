"""WHS-R2: residual layer routing that keeps text-adaptivity alive.

Successor to :class:`ALAEMotionGPTWeightedHS`. The static WHS router
collapsed to a text-independent per-slot layer pick: Gumbel noise suppression
inflated the logit margins to 17-23, saturating the softmax and killing the
gradient through any text-conditional structure (which demonstrably existed
at ep49 on slot k2 and survives only as a +-2 logit "dark structure").

Four changes, each aimed at that failure mode:

1. RESIDUAL ROUTING with bounded parts:
       logits = s_s * tanh(static / s_s) + s_d * tanh(delta(text) / s_d)
   `static` is a free (K, L) parameter — the collapsed solution lives here,
   so the model never has to pay for adaptivity it does not use (performance
   floor = the static WHS solution). `delta(text)` is the per-text correction;
   both parts are tanh-capped so NO margin can grow past s_s + s_d and the
   softmax can never saturate away the gradient. The delta head's last layer
   is zero-initialised: training starts exactly at the (learnable) static
   solution.

2. FULL-TOKEN MEMORY: the parent pooled the prompt into ONE memory token,
   degenerating cross-attention into a gated linear map. Here the memory is
   the whole last-layer token sequence (projected H->D) with the pad mask
   passed to the attention, so queries can attend to the action verbs.

3. MUTUAL-INFORMATION REGULARISER (per slot, batch-level):
       L_mi = E_text[H(p)] - H(E_text[p]),   p = softmax(logits / tau)
   i.e. push conditional entropy DOWN (each text routes decisively) and
   marginal entropy UP (different texts route differently) — the Switch-
   Transformer load-balancing idea recast as maximising I(route; text).
   NB: naive entropy bonuses are wrong here — they force soft mixtures for
   every text instead of conditional switching.

4. Slower/warmer tau: configs pair this class with gumbel_tau_end 1.5 and
   gumbel_anneal_epochs 100 (vs 0.5/50) so exploration survives past the
   epoch range (49-199) where the static version locked in.

Diagnostics: train/route_mi, train/route_cond_ent, train/route_marg_ent join
the parent's router_entropy/tau logs. `_last_router_logits` carries the
COMPOSED logits for analysis scripts (the layer_router hook now only sees
the raw delta).

Optional LOAD-BALANCING regulariser (OFF by default):
    L_lb = mean_k KL(p_bar_k || Uniform(L)) = mean_k [ln L - H(p_bar_k)]
Motivation: on HML K4 slots k0/k1/k3 stayed 100% static on L10 (only k2
routes by text). For an already-collapsed slot the MI term above sits at a
zero-cost saddle (marg_ent = cond_ent = 0), so its outward pressure is weak.
The KL-to-uniform has a non-zero gradient AT collapse and vanishes only when
the batch marginal is uniform. Combined with the MI term this is an
"asymmetric MI": marginal-entropy weight = lambda_route_mi + lambda_route_lb,
conditional-entropy weight = lambda_route_mi (per-sample decisiveness still
guarded). Known cheat path: per-sample uniform SOFT mixtures also satisfy the
marginal — screening must watch train/route_cond_ent for that escape.
lambda_route_lb = 0.0 keeps behaviour byte-identical to plain R2. Extra
diagnostics: train/route_lb, per-slot train/route_marg_ent_k{i}.

Optional SLOT-SEPARATION regulariser (OFF by default):
    L_sep = mean_{k<k'} <p_bar_k, p_bar_k'>
i.e. the mean pairwise dot-product OVERLAP of the batch-marginal routing
distributions across slots, minimised. Motivation: the LB line showed that fighting per-slot CONFIDENCE (KL to uniform forces every slot
to spread over layers) unlocks all slots but costs test FID. What we
actually want is slots occupying DIFFERENT layers; L_sep only penalises two
slots sharing the same layers and is exactly zero for confident-but-disjoint
slots (e.g. four distinct one-hots) — confidence itself is never taxed, so
the LB cheat pressure toward per-sample soft mixtures has no analogue here.
At the collapsed state (k0/k1/k3 all one-hot on L10) the k-k' overlaps are 1
with a non-zero gradient pushing the tied slots apart. Known cheat path: a
slot can lower overlap by thinning its marginal across many layers — watch
train/route_marg_ent_k{i} (should stay LOW) and train/route_argmax_k{i}
(should become pairwise DISTINCT) to tell separation from spreading.
lambda_route_sep = 0.0 keeps behaviour byte-identical to plain R2. Extra
diagnostics: train/route_sep, per-slot train/route_argmax_k{i}.

Slot-symmetry breaking (both OFF by default). A per-sample clone check
showed k0/k1/k3 make ONE
shared text-dependent decision copied three times (per-sample argmax
agreement .98 vs .41 expected under independence in LB05) and that BOTH
prior regularisers are marginal functionals which are mathematically blind
to per-sample cloning. Two remedies:

1. router_layer_bands (list of K entries, each [lo, hi] inclusive or null):
   hard per-slot layer bands; logits outside a slot's band get -1e4 before
   any softmax. Disjoint bands make cloning IMPOSSIBLE by construction while
   leaving within-band routing free to be text-adaptive. null = free slot.
2. lambda_route_sep_ps: PER-SAMPLE pairwise overlap
       L_sep_ps = E_x[mean_{k<k'} <p_k(x), p_k'(x)>]
   minimised — unlike the marginal L_sep this SEES cloning (two slots
   choosing the same layer for the same caption). Pair with the MI term's
   cond_ent pressure to guard the smearing escape.

Extra diagnostics: train/route_sep_ps (when enabled) and
train/route_clone_agree (always; mean per-sample argmax agreement over slot
pairs — the clone observable; K=4 clones sit ~0.5 = 3 locked pairs of 6,
de-cloned ~0.02).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from motGPT.models.alae_motion_gpt_weighted_hs import ALAEMotionGPTWeightedHS
from adaptive_length_auto_encoder.model import CrossAttentionBlock


class MaskedCrossAttentionBlock(CrossAttentionBlock):
    """Parent block + key_padding_mask on the cross-attention (True = pad)."""

    def forward(self, target, memory, memory_key_padding_mask=None):
        x = self.norm1(target)
        x_sa, _ = self.self_attn(x, x, x, need_weights=False)
        target = target + self.drop1(x_sa)

        x = self.norm2(target)
        x_ca, _ = self.cross_attn(
            x, memory, memory, need_weights=False,
            key_padding_mask=memory_key_padding_mask)
        target = target + self.drop2(x_ca)

        x = self.norm3(target)
        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        target = target + self.drop3(x_ff)
        return target


class ALAEMotionGPTWeightedHSR2(ALAEMotionGPTWeightedHS):
    """Residual, saturation-proof, text-adaptive layer router."""

    def __init__(
        self,
        *args,
        router_static_scale: float = 4.0,
        router_delta_scale: float = 4.0,
        lambda_route_mi: float = 0.05,
        lambda_route_lb: float = 0.0,
        lambda_route_sep: float = 0.0,
        lambda_route_sep_ps: float = 0.0,
        router_layer_bands=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.router_static_scale = float(router_static_scale)
        self.router_delta_scale = float(router_delta_scale)
        self.lambda_route_mi = float(lambda_route_mi)
        self.lambda_route_lb = float(lambda_route_lb)
        self.lambda_route_sep = float(lambda_route_sep)
        self.lambda_route_sep_ps = float(lambda_route_sep_ps)

        K = int(self.hparams.k_motion_tokens)
        alae_cfg = dict(kwargs.get('alae') or {})
        nhead = int(alae_cfg.get('nhead', 8))
        dim_feedforward = int(alae_cfg.get('dim_feedforward', 1024))
        dropout = float(alae_cfg.get('dropout', 0.1))
        activation = str(alae_cfg.get('activation', 'gelu'))
        D = self.alae_wrapper.latent_dim

        # Free static routing table (the collapsed solution's new home).
        self.static_router_logits = nn.Parameter(
            torch.zeros(K, self.n_layers))

        # Hard per-slot layer bands (2026-07-20, see module docstring).
        # Non-persistent buffer: rebuilt from hparams, so checkpoints stay
        # interoperable with band-free runs.
        if router_layer_bands is not None:
            bands = [None if b is None else (int(b[0]), int(b[1]))
                     for b in list(router_layer_bands)]
            assert len(bands) == K, (
                f'router_layer_bands needs {K} entries, got {len(bands)}')
            mask = torch.zeros(K, self.n_layers)
            for k, band in enumerate(bands):
                if band is None:
                    continue
                lo, hi = band
                assert 0 <= lo <= hi < self.n_layers, f'bad band {band} for k{k}'
                mask[k, :lo] = -1e4
                mask[k, hi + 1:] = -1e4
            self.register_buffer('router_band_mask', mask, persistent=False)
        else:
            self.router_band_mask = None

        # Mask-aware cross-attn stack replaces the parent's (same shapes, so
        # checkpoints interop; the parent's blocks are discarded).
        self.cross_attn_blocks = nn.ModuleList(
            [
                MaskedCrossAttentionBlock(
                    d_model=D,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.num_cross_attn_layers)
            ]
        )

        # Zero-init the delta head's output layer: training starts EXACTLY at
        # the (learnable) static solution; the text branch must earn its way in.
        nn.init.zeros_(self.layer_router[-1].weight)
        nn.init.zeros_(self.layer_router[-1].bias)

        self._last_router_logits = None
        self._last_route_mi = None
        self._last_route_lb = None
        self._route_marg_ent_slots = None
        self._last_route_sep = None
        self._route_argmax_slots = None
        self._last_route_sep_ps = None
        self._route_clone_agree = None

    # ------------------------------------------------------------------
    def _encode_text_ar(self, texts: List[str]) -> torch.Tensor:
        """Parent rollout + full-token memory + residual bounded logits."""
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

        out = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            return_dict=True,
        )

        # Branch A memory: FULL last-layer token sequence (B, T, D) + pad mask
        # (the parent pooled this to a single token, which starves the router
        # of the very tokens - action verbs - it should route on).
        last_layer = out.hidden_states[-1]                      # (B, T, H)
        memory = self.text_mem_proj(last_layer)                 # (B, T, D)
        pad_mask = attention_mask == 0                          # True = pad

        def _stack_all_layers(hidden_states) -> torch.Tensor:
            cols = [h[:, -1:, :] for h in hidden_states[1:]]
            return torch.stack(cols, dim=2)                     # (B, 1, L, H)

        all_steps = [_stack_all_layers(out.hidden_states)]
        last_hiddens = [out.hidden_states[-1][:, -1:, :]]
        past = out.past_key_values if use_cache else None
        cur_attn = attention_mask
        prompt_embeds = None

        for _ in range(K - 1):
            next_input = self.feedback_norm(last_hiddens[-1])
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
                cat_h = torch.cat(last_hiddens, dim=1)
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

        H_all = torch.cat(all_steps, dim=1)                     # (B, K, L, H)
        B = H_all.shape[0]

        latents = self.stage2_latent_queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.cross_attn_blocks:
            latents = block(latents, memory, memory_key_padding_mask=pad_mask)
        latents = self.cross_attn_norm(latents)                 # (B, K, D)

        # Residual bounded logits: neither part can saturate the softmax.
        s_s, s_d = self.router_static_scale, self.router_delta_scale
        static = s_s * torch.tanh(self.static_router_logits / s_s)   # (K, L)
        raw_delta = self.layer_router(latents)                       # (B, K, L)
        delta = s_d * torch.tanh(raw_delta.float() / s_d)
        logits = static.unsqueeze(0).float() + delta                 # (B, K, L)

        # Hard per-slot layer bands (2026-07-20): applied BEFORE every softmax
        # so weights, the MI/LB/SEP statistics and all diagnostics see the
        # banded distribution. None = byte-identical to plain R2.
        if self.router_band_mask is not None:
            logits = logits + self.router_band_mask.unsqueeze(0)

        if self.training:
            tau = self._gumbel_tau()
            weights = F.gumbel_softmax(
                logits, tau=tau, hard=False, dim=-1).to(H_all.dtype)
        else:
            tau = self._router_infer_tau()
            weights = F.softmax(logits / tau, dim=-1).to(H_all.dtype)

        weighted = torch.einsum('bkl,bklh->bkh', weights, H_all)

        if self.stage2_latent_queries.requires_grad:
            from adaptive_length_auto_encoder.model import (
                latent_query_orthogonality_loss,
            )
            self._last_ortho = latent_query_orthogonality_loss(self.stage2_latent_queries)
        else:
            self._last_ortho = None

        # MI regulariser on the NOISE-FREE conditional p = softmax(logits/tau)
        # (Gumbel noise would inflate the conditional entropy artificially).
        # Differentiable; only applied on train steps via forward_motion.
        p = F.softmax(logits / tau, dim=-1)                     # (B, K, L)
        cond_ent = -(p.clamp_min(1e-9).log() * p).sum(-1).mean(0)      # (K,)
        p_bar = p.mean(0)                                       # (K, L)
        marg_ent = -(p_bar.clamp_min(1e-9).log() * p_bar).sum(-1)      # (K,)
        self._last_route_mi = (marg_ent - cond_ent).mean()      # maximise
        self._route_ent_stats = (cond_ent.detach().mean(), marg_ent.detach().mean())

        # Load-balancing regulariser (2026-07-14, see module docstring):
        # KL(p_bar_k || uniform) per slot, minimised. Non-zero gradient at
        # collapse (unlike MI, whose collapsed state is a zero-cost saddle);
        # zero only at a uniform batch marginal. Slot-uniform on purpose: an
        # already-spread slot contributes ~0, no slot indices are hardcoded.
        self._last_route_lb = (math.log(p.size(-1)) - marg_ent).mean()
        self._route_marg_ent_slots = marg_ent.detach()

        # Slot-separation regulariser (2026-07-19, see module docstring):
        # mean pairwise overlap <p_bar_k, p_bar_k'> across slots, minimised.
        # Zero for confident-but-disjoint slots; gradient pushes slots that
        # share layers apart without ever taxing per-slot confidence.
        overlap = p_bar @ p_bar.t()                             # (K, K)
        iu = torch.triu_indices(overlap.size(0), overlap.size(1), offset=1)
        self._last_route_sep = overlap[iu[0], iu[1]].mean()
        self._route_argmax_slots = p_bar.detach().argmax(dim=-1)

        # PER-SAMPLE slot-separation (2026-07-20, see module docstring): the
        # marginal L_sep above is blind to cloning (three slots copying one
        # text-dependent choice have near-uniform marginals with LOW marginal
        # overlap); this one penalises two slots picking the same layers FOR
        # THE SAME caption. Only computed when enabled (default path untouched).
        if self.lambda_route_sep_ps != 0.0:
            ov_ps = torch.einsum('bkl,bjl->bkj', p, p)          # (B, K, K)
            self._last_route_sep_ps = ov_ps[:, iu[0], iu[1]].mean()
        else:
            self._last_route_sep_ps = None

        # Clone observable (always on, no_grad): mean per-sample argmax
        # agreement over slot pairs. K=4 clones ~0.5 (3 locked pairs of 6),
        # independent slots ~their marginal collision rate (~0.02 de-cloned).
        with torch.no_grad():
            am = p.argmax(dim=-1)                               # (B, K)
            self._route_clone_agree = (
                am[:, iu[0]] == am[:, iu[1]]).float().mean()

        with torch.no_grad():
            w = weights.float().clamp_min(1e-9)
            entropy = -(w * w.log()).sum(dim=-1).mean()
            self._last_router_stats = {
                'entropy': entropy,
                'tau': entropy.new_tensor(float(tau)),
            }
        self._last_router_weights = weights.detach()
        self._last_router_logits = logits.detach()

        return weighted

    # ------------------------------------------------------------------
    def forward_motion(self, batch):
        out = super().forward_motion(batch)          # adds ortho if enabled
        mi = self._last_route_mi
        if mi is not None and self.lambda_route_mi != 0.0 and self.training:
            out['loss'] = out['loss'] - self.lambda_route_mi * mi
            out['loss_route_mi'] = -mi.detach()
        lb = self._last_route_lb
        if lb is not None and self.lambda_route_lb != 0.0 and self.training:
            out['loss'] = out['loss'] + self.lambda_route_lb * lb
            out['loss_route_lb'] = lb.detach()
        sep = self._last_route_sep
        if sep is not None and self.lambda_route_sep != 0.0 and self.training:
            out['loss'] = out['loss'] + self.lambda_route_sep * sep
            out['loss_route_sep'] = sep.detach()
        sps = self._last_route_sep_ps
        if sps is not None and self.lambda_route_sep_ps != 0.0 and self.training:
            out['loss'] = out['loss'] + self.lambda_route_sep_ps * sps
            out['loss_route_sep_ps'] = sps.detach()
        return out

    def allsplit_step(self, split: str, batch, batch_idx):
        result = super().allsplit_step(split, batch, batch_idx)
        if split == 'train' and self._last_route_mi is not None:
            cond_ent, marg_ent = self._route_ent_stats
            batch_size = len(batch['text'])
            logs = [
                ('train/route_mi', self._last_route_mi.detach()),
                ('train/route_cond_ent', cond_ent),
                ('train/route_marg_ent', marg_ent),
            ]
            if self._last_route_lb is not None:
                logs.append(('train/route_lb', self._last_route_lb.detach()))
            if self._last_route_sep is not None:
                logs.append(('train/route_sep', self._last_route_sep.detach()))
            if self._last_route_sep_ps is not None:
                logs.append(('train/route_sep_ps',
                             self._last_route_sep_ps.detach()))
            if self._route_clone_agree is not None:
                # THE clone observable: per-sample argmax agreement over slot
                # pairs (see 2026-07-20 docstring section).
                logs.append(('train/route_clone_agree',
                             self._route_clone_agree))
            if self._route_argmax_slots is not None:
                # Per-slot argmax layer of the batch-marginal routing: THE
                # observable for "do the four slots sit on DISTINCT layers"
                # (marg_ent misses a slot moving wholesale L10 -> L9).
                logs.extend(
                    (f'train/route_argmax_k{i}', v.float())
                    for i, v in enumerate(self._route_argmax_slots))
            if self._route_marg_ent_slots is not None:
                # Per-slot marginal entropy: THE observable for "did k0/k1/k3
                # leave L10" (the K-mean above hides single-slot moves).
                logs.extend(
                    (f'train/route_marg_ent_k{i}', v)
                    for i, v in enumerate(self._route_marg_ent_slots))
            for name, val in logs:
                self.log(name, val, on_step=False, on_epoch=True,
                         prog_bar=False, logger=True, sync_dist=True,
                         batch_size=batch_size)
        return result
