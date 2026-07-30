import os
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from motGPT.losses.wing_loss import wing_loss
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, AutoModelForCausalLM
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 256,
        lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage
        self.sample_temperature = kwargs.get('sample_temperature', 0.9)
        self.sample_top_p = kwargs.get('sample_top_p', 0.95)
        # Wing Loss hyperparameters used by PredLoss / decoder_only_loss.
        # Defaults (10.0, 2.0) match the original implementation.
        self.wing_w = float(kwargs.get('wing_w', 10.0))
        self.wing_eps = float(kwargs.get('wing_eps', 2.0))
        # Forced-length AR generation: when True, callers supplying a per-sample
        # target_token_lengths list will get exactly that many codebook tokens
        # followed by an EOM (via a step-aware prefix_allowed_tokens_fn).
        # Useful at eval time to force AR output to match GT token length.
        self.force_length = bool(kwargs.get('force_length', False))

        # Instantiate language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            self.language_model = T5ForConditionalGeneration.from_pretrained(
                model_path)
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = 'dec'
        elif model_type == "llama":
            self.language_model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use float32 for training stability
                low_cpu_mem_usage=True
            )
            self.lm_type = 'dec'
        else:
            # Generic decoder-only backbone: qwen / qwen2 / qwen3, mistral,
            # gemma, phi, ... AutoModelForCausalLM resolves the concrete
            # architecture from the checkpoint's config.json, so any HF causal
            # LM works here without a dedicated branch. (Qwen3 needs
            # transformers >= 4.51.)
            self.language_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # Use float32 for training stability
                low_cpu_mem_usage=True,
                trust_remote_code=True,  # Required by some custom architectures
            )
            self.lm_type = 'dec'

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'  # Required for decoder-only models

        # Add motion tokens
        self.tokenizer.add_tokens(
            [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)])

        if new_token_type == "insert":
            old_vocab_size = self.language_model.get_input_embeddings().weight.shape[0]
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            
            # Initialize new embeddings properly (avoid NaN in float16)
            with torch.no_grad():
                embed_layer = self.language_model.get_input_embeddings()
                new_vocab_size = embed_layer.weight.shape[0]
                if new_vocab_size > old_vocab_size:
                    # Get mean and std from existing embeddings (in float32 for stability)
                    existing_emb = embed_layer.weight[:old_vocab_size].float()
                    mean_val = existing_emb.mean().item()
                    std_val = existing_emb.std().item()
                    
                    # Initialize new tokens with similar distribution
                    new_emb = torch.randn(
                        new_vocab_size - old_vocab_size, 
                        embed_layer.weight.shape[1],
                        device=embed_layer.weight.device,
                        dtype=torch.float32
                    ) * std_val + mean_val
                    
                    # Convert to target dtype and assign
                    embed_layer.weight[old_vocab_size:] = new_emb.to(embed_layer.weight.dtype)
                    
                    # Also handle lm_head if it exists and is separate
                    lm_head = self.language_model.get_output_embeddings()
                    if lm_head is not None and lm_head.weight is not embed_layer.weight:
                        lm_head.weight[old_vocab_size:] = new_emb.to(lm_head.weight.dtype)
                        
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.m_codebook_size + 3)
            # lm_head = NewTokenEmb(self.language_model.lm_head,
            #   self.m_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared
            # self.language_model.lm_head = lm_head

        # Some decoder-side adapters consume LLM hidden states directly with no
        # latent projector in between, so their input dim must match
        # language_model.hidden_size.

        # LoRA (parameter-efficient fine-tuning). Hyperparameters are
        # config-driven so each backbone can pick its own rank and target
        # modules. NOTE: peft has no default `target_modules` mapping for some
        # newer backbones (e.g. qwen3 -> None in peft 0.15.x), so those configs
        # MUST pass `lora_target_modules` explicitly or get_peft_model raises.
        if lora:
            from peft import LoraConfig, get_peft_model
            target_modules = (list(lora_target_modules)
                              if lora_target_modules is not None else None)
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=target_modules,
            )
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)

    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict, output_hidden_states: bool = False,
                **kwargs):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks, output_hidden_states)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks, output_hidden_states,
                                    **kwargs)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def _build_single_seed_supervised_texts(self,
                                            texts: List[str],
                                            lengths: List[int],
                                            tasks: Optional[List[dict]] = None):
        """Build decoder-only supervised strings for a single motion seed token."""
        if tasks is None:
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        motion_strings = [''] * len(texts)
        inputs, _ = self.template_fulfill(tasks, lengths, motion_strings, texts)
        seed_token = f'<motion_id_{self.m_codebook_size}>'
        labels = [inputs[i] + ' \n ' + seed_token + self.tokenizer.eos_token
                  for i in range(len(inputs))]
        return inputs, labels, seed_token

    def extract_single_seed_hidden(self,
                                   texts: List[str],
                                   lengths: List[int],
                                   tasks: Optional[List[dict]] = None,
                                   compute_loss: bool = False):
        """Extract the last-layer hidden state aligned to a single motion seed token.

        The seed token is the motion start marker `<motion_id_{codebook_size}>`.
        We use the causal-shifted hidden state whose next-token label is that
        seed token, yielding a single `(B, 1, hidden_dim)` hidden seed.
        """
        if self.lm_type != 'dec':
            raise NotImplementedError('single-seed hidden extraction currently supports decoder-only LMs only')

        self.tokenizer.padding_side = 'right'
        _, labels, seed_token = self._build_single_seed_supervised_texts(texts, lengths, tasks)

        enc = self.tokenizer(labels,
                             padding='max_length',
                             max_length=self.max_length,
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors='pt')

        input_ids = enc.input_ids.to(self.language_model.device)
        attention_mask = enc.attention_mask.to(self.language_model.device)

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, T, hidden_dim)

        seed_token_id = self.tokenizer.convert_tokens_to_ids(seed_token)
        shift_hidden = last_hidden[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        seed_hiddens = []
        for i in range(input_ids.shape[0]):
            seed_positions = (shift_labels[i] == seed_token_id).nonzero(as_tuple=True)[0]
            if len(seed_positions) == 0:
                raise RuntimeError(f'No single seed token found in sample {i}')
            seed_hidden = shift_hidden[i, seed_positions[0]:seed_positions[0] + 1, :]
            seed_hiddens.append(seed_hidden)

        seed_hidden_batch = torch.stack(seed_hiddens, dim=0)

        if compute_loss:
            labels_for_loss = input_ids.clone()
            labels_for_loss[labels_for_loss == self.tokenizer.pad_token_id] = -100
            labels_for_loss[:, :-1] = -100
            logits = outputs.logits[..., :-1, :].contiguous()
            shifted = labels_for_loss[..., 1:].contiguous()
            valid = shifted != -100
            if valid.any():
                outputs.loss = F.cross_entropy(logits[valid], shifted[valid])
            else:
                outputs.loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            outputs.loss = torch.tensor(0.0, device=input_ids.device)

        return seed_hidden_batch, outputs

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
        output_hidden_states: bool = False,
    ):

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(
            motion_tokens.device)
        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)

        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(
                mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(
                target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids,
                                                     target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids,
                                                     input_ids_sentinel)

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = target_inputs.attention_mask.to(
                motion_tokens.device)

        labels_input_ids[labels_input_ids == 0] = -100
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,
            labels=labels_input_ids,
            decoder_attention_mask=lables_attention_mask
            if condition == 'supervised' else None,
        )

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
        output_hidden_states: bool = False,
        vae=None,
        gt_motion=None,
        motion_lengths=None,
        gt_residual=None,
        gt_token_lengths=None,
    ):
        self.tokenizer.padding_side = "right"

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Always use supervised (t2m) condition
        condition = 'supervised'

        inputs, outputs = self.template_fulfill(tasks, lengths,
                                                motion_strings, texts)
        labels = []
        for i in range(len(inputs)):
            labels.append(inputs[i] + ' \n ' + outputs[i] +
                          self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        labels_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        
        # =====================================================
        # CRITICAL FIX: Mask text portion - only compute loss on motion tokens
        # This ensures the model learns text->motion mapping, not just language modeling
        # =====================================================
        labels_for_loss = labels_input_ids.clone()
        
        # Mask padding tokens
        labels_for_loss[labels_for_loss == self.tokenizer.pad_token_id] = -100
        
        # Mask text portion - only compute loss on motion tokens
        # Get SOM token id (start of motion)
        som_id = self.tokenizer.convert_tokens_to_ids(f'<motion_id_{self.m_codebook_size}>')
        
        # For each sample, find SOM position and mask everything before it
        batch_size = labels_input_ids.shape[0]
        for i in range(batch_size):
            # Find SOM position
            som_positions = (labels_input_ids[i] == som_id).nonzero(as_tuple=True)[0]
            if len(som_positions) > 0:
                som_pos = som_positions[0].item()
                # Mask all tokens before SOM (SOM itself is NOT masked by [:som_pos])
                labels_for_loss[i, :som_pos] = -100
        
        # Force output_hidden_states when computing motion recon losses
        need_hidden = (vae is not None and gt_motion is not None and motion_lengths is not None)
        effective_output_hidden_states = output_hidden_states or need_hidden

        # Forward WITHOUT built-in label loss — we compute a custom
        # motion-restricted cross-entropy below.
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=labels_attention_mask,
                                      output_hidden_states=effective_output_hidden_states)

        # === Motion-only cross-entropy loss ===
        # Train only on motion-related tokens (512 codebook + SOM/EOM/MASK);
        # the 50k+ text vocab is completely excluded from the output softmax
        # so the LM head only learns to rank motion tokens against each other.
        logits = outputs.logits                          # (B, T, V)
        shift_logits = logits[..., :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels_for_loss[..., 1:].contiguous()  # (B, T-1)

        motion_tok_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
        motion_tok_end   = motion_tok_start + self.m_codebook_size + 3  # exclusive, includes SOM/EOM/MASK

        is_motion = (shift_labels >= motion_tok_start) & (shift_labels < motion_tok_end)
        n_motion = is_motion.sum().item()

        if n_motion > 0:
            # Softmax restricted to the 515 motion tokens — text vocab is ignored.
            mot_logits = shift_logits[is_motion][:, motion_tok_start:motion_tok_end]
            mot_labels = shift_labels[is_motion] - motion_tok_start
            outputs.loss = F.cross_entropy(mot_logits, mot_labels)
        else:
            outputs.loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute motion reconstruction losses using last-layer hidden states
        if need_hidden:
            last_hidden = outputs.hidden_states[-1]  # (B, T, hidden_dim)
            loss_dict = self._compute_motion_recon_losses(
                last_hidden, labels_input_ids, vae, gt_motion, motion_lengths,
                gt_residual=gt_residual, gt_token_lengths=gt_token_lengths)
            outputs.pred_loss = loss_dict['pred_loss']
            outputs.rec_loss = loss_dict['rec_loss']
            outputs.decoder_only_loss = loss_dict['decoder_only_loss']
            outputs.residual_pred_norm = loss_dict['residual_pred_norm']
            outputs.residual_kd_loss = loss_dict['reskd_loss']

        return outputs

    @torch.no_grad()
    def forward_with_hidden(self,
                            texts: List[str],
                            motion_tokens: Tensor,
                            lengths: List[int],
                            tasks: Optional[List[dict]] = None):
        """Teacher-forced forward pass: extract LLM hidden states aligned to GT motion tokens.

        Used by Stage 3 fine-tuning to cache (hidden, codes, motion) triples where
        codes are the GT token ids (not LLM-sampled), so the residual predictor sees
        the true distribution the LLM will be asked to emit at inference time while
        the rest of the pipeline stays teacher-forced.

        Pipeline:
          1. Build supervised prompt: "caption \\n <som><m_tok_0>...<m_tok_N><eom><eos>"
          2. Single LM forward with output_hidden_states=True (no_grad).
          3. Causal shift: hidden[t] corresponds to the prediction of token[t+1];
             keep positions where token[t+1] is a codebook token.
          4. Return per-sample hidden (N_mot_i, hidden_dim) and code ids (N_mot_i,).

        Args:
            texts:         list of captions (length B)
            motion_tokens: (B, T_mot_max) GT codebook indices in [0, codebook_size)
            lengths:       list of int, per-sample true motion token count
            tasks:         list[B] of templates as in train_lm_forward; if None, uses
                           a minimal caption->motion template identical to generation.

        Returns:
            hidden_per_sample: list of (N_mot_i, hidden_dim) float tensors
            codes_per_sample:  list of (N_mot_i,) long tensors (values in [0, codebook_size))
        """
        self.tokenizer.padding_side = "right"
        device = motion_tokens.device

        # Build supervised prompt identical to forward_dec's training template.
        if tasks is None:
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['<Motion_Placeholder>']
            }] * len(texts)

        motion_strings = self.motion_token_to_string(motion_tokens, lengths)
        inputs_str, outputs_str = self.template_fulfill(tasks, lengths,
                                                        motion_strings, texts)
        labels = [inputs_str[i] + ' \n ' + outputs_str[i] + self.tokenizer.eos_token
                  for i in range(len(inputs_str))]

        enc = self.tokenizer(labels,
                             padding='max_length',
                             max_length=self.max_length,
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)

        fwd_out = self.language_model(input_ids=input_ids,
                                      attention_mask=attn_mask,
                                      output_hidden_states=True)
        last_hidden = fwd_out.hidden_states[-1]  # (B, T, hidden_dim)

        motion_id_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
        motion_id_end = motion_id_start + self.m_codebook_size

        # Causal shift: hidden[t] predicts token[t+1]
        shift_hidden = last_hidden[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        is_codebook = (shift_ids >= motion_id_start) & (shift_ids < motion_id_end)

        hidden_per_sample = []
        codes_per_sample = []
        for i in range(input_ids.shape[0]):
            mask_i = is_codebook[i]
            if mask_i.sum().item() > 0:
                hidden_i = shift_hidden[i, mask_i].contiguous()                  # (N_mot, hidden_dim)
                code_idx_i = (shift_ids[i, mask_i] - motion_id_start).contiguous()  # (N_mot,)
                hidden_per_sample.append(hidden_i)
                codes_per_sample.append(code_idx_i)
            else:
                hidden_per_sample.append(None)
                codes_per_sample.append(None)

        return hidden_per_sample, codes_per_sample

    def _compute_motion_recon_losses(self, hidden_states, labels_input_ids, vae, gt_motion, motion_lengths,
                                     gt_residual=None, gt_token_lengths=None):
        """Compute PredLoss using hidden-state latent projection.

        Pipeline (per sample):
          1. Extract last-layer hidden states at motion-token positions (causal-shifted)
          2. Feed hidden states directly into residual_predictor (via vae.decode_from_latent)
          3. PredLoss: vae.decode_from_latent(hidden_input, code_idx)
             Discrete feature + predicted residual → (optional length aligner)
             → decoder → MSE vs GT motion

        PredLoss is differentiable — gradients flow through:
          decoder ← length_aligner ← residual_predictor ← LLM hidden states

        Also computes decomposed component losses for monitoring:
          - decoder_only_loss: decode from codewords only (no residual) → MSE vs GT
          - residual_pred_norm: average L2 norm of predicted residuals

        Args:
            hidden_states: (B, T, hidden_dim) last-layer hidden states from LLM
            labels_input_ids: (B, T) original token IDs (before loss masking)
            vae: motion decoder / autoencoder wrapper instance
            gt_motion: (B, T_max, nfeats) ground truth motion features
            motion_lengths: list of int, actual motion lengths
            gt_residual: optional precomputed encoder residual (B, T_tok, code_dim)
            gt_token_lengths: optional list/tensor of per-sample GT motion
                token counts (typically ``batch['m_tokens_len']``). When the
                VAE has a length aligner configured, this is forwarded as
                ``target_lengths`` so the predicted-token sequence is
                losslessly re-mapped to the GT token length before decoding.

        Returns:
            dict with keys: pred_loss, rec_loss, decoder_only_loss, residual_pred_norm
        """
        import torch.nn.functional as F

        device = hidden_states.device

        # Motion codebook token ID range in the vocabulary
        motion_id_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
        motion_id_end = motion_id_start + self.m_codebook_size

        # Causal shift: hidden_states[:, t, :] predicts labels[:, t+1]
        shift_hidden = hidden_states[:, :-1, :]   # (B, T-1, hidden_dim)
        shift_labels = labels_input_ids[:, 1:]      # (B, T-1)

        # Mask for positions that predict codebook tokens
        is_codebook = (shift_labels >= motion_id_start) & (shift_labels < motion_id_end)

        batch_size = hidden_states.shape[0]
        pred_losses = []
        decoder_only_losses = []
        residual_norms = []
        residual_kd_losses = []

        # Teacher signal for residual_predictor: encoder's continuous residual_delta.
        # Prefer the precomputed GT residual from the batch (no compute cost),
        # fall back to on-the-fly encoding if not provided.
        if gt_residual is not None:
            gt_residual_batch = gt_residual.to(hidden_states.device)
        else:
            with torch.no_grad():
                _, gt_residual_batch = vae.encode_with_residual(gt_motion)  # (B, T_tok, code_dim)

        for i in range(batch_size):
            mask_i = is_codebook[i]
            n_mot = mask_i.sum().item()
            if n_mot == 0:
                continue

            # Extract hidden states at motion positions — fed directly to residual_predictor
            hidden_i = shift_hidden[i, mask_i]  # (N_mot, hidden_dim)

            # Extract codebook indices from GT labels (token_id - motion_id_start)
            token_ids_i = shift_labels[i, mask_i]  # (N_mot,)
            code_idx_i = token_ids_i - motion_id_start  # (N_mot,) codebook indices [0, codebook_size)

            # GT motion for this sample
            length_i = motion_lengths[i]
            gt_i = gt_motion[i:i+1, :length_i, :]

            # Resolve target token length for the length aligner. When
            # gt_token_lengths is supplied AND the VAE has a length aligner,
            # the latent will be re-mapped from N_pred=n_mot to N_gt=n_gt_i
            # before the motion decoder.
            decode_kwargs = {}
            if gt_token_lengths is not None and getattr(vae, 'has_length_aligner', False):
                if torch.is_tensor(gt_token_lengths):
                    n_gt_i = int(gt_token_lengths[i].item())
                else:
                    n_gt_i = int(gt_token_lengths[i])
                if n_gt_i > 0:
                    decode_kwargs['target_lengths'] = [n_gt_i]
                    decode_kwargs['memory_lengths'] = [int(n_mot)]

            # --- PredLoss: dual-path decode (hidden states → residual_predictor + codebook codewords) ---
            decoded_pred, intermediates = vae.decode_from_latent(
                hidden_i.unsqueeze(0),
                code_idx=code_idx_i.unsqueeze(0),
                return_intermediates=True,
                **decode_kwargs)  # (1, T_motion, nfeats)

            min_len = min(decoded_pred.shape[1], gt_i.shape[1])
            pred_loss_i = wing_loss(decoded_pred[:, :min_len, :], gt_i[:, :min_len, :],
                                    w=self.wing_w, eps=self.wing_eps)
            pred_losses.append(pred_loss_i)

            # --- Decoder-only loss: decode from codewords without residual ---
            codewords = intermediates['codewords']  # (1, T_tok, code_dim)
            x_d_only = codewords.permute(0, 2, 1).contiguous()  # (1, code_dim, T_tok)
            x_d_only = F.normalize(x_d_only, dim=1) * vae.rms_alpha
            with torch.no_grad():
                decoded_only = vae.postprocess(vae.vqvae.decoder(x_d_only))
            decoder_only_loss_i = wing_loss(decoded_only[:, :min_len, :], gt_i[:, :min_len, :],
                                            w=self.wing_w, eps=self.wing_eps)
            decoder_only_losses.append(decoder_only_loss_i)

            # --- Residual prediction norm ---
            pred_res = intermediates['pred_residual']  # (1, T_tok, code_dim)
            residual_norms.append(pred_res.norm(dim=-1).mean())

            # --- Residual KD loss: align predicted residual with encoder's GT residual ---
            # pred_res aligns with LLM motion positions (n_mot tokens). The batch-level
            # gt_residual is padded to max token length across the batch; slice the
            # first n_mot entries for this sample.
            n_mot = pred_res.shape[1]
            assert gt_residual_batch.dim() == 3 and gt_residual_batch.shape[-1] == pred_res.shape[-1], (
                f"gt_residual shape mismatch: got {tuple(gt_residual_batch.shape)}, "
                f"expected (B, T_tok, {pred_res.shape[-1]})")
            assert gt_residual_batch.shape[1] >= n_mot, (
                f"gt_residual token length {gt_residual_batch.shape[1]} shorter than "
                f"LLM motion positions {n_mot}; check dataset/collate padding")
            gt_res_i = gt_residual_batch[i:i+1, :n_mot, :]
            if n_mot > 0:
                # Scale-invariant relative MSE: = 1 when pred_res=0, = 0 when pred_res=gt.
                # Makes the KD term comparable across residual magnitudes and usable with
                # lambda≈1.0 instead of requiring ~1000x scaling for raw MSE.
                num = F.mse_loss(pred_res, gt_res_i, reduction='mean')
                denom = gt_res_i.detach().pow(2).mean().clamp_min(1e-8)
                kd_loss_i = num / denom
                residual_kd_losses.append(kd_loss_i)

        if pred_losses:
            pred_loss = torch.stack(pred_losses).mean()
            decoder_only_loss = torch.stack(decoder_only_losses).mean()
            residual_pred_norm = torch.stack(residual_norms).mean()
        else:
            pred_loss = torch.tensor(0., device=device)
            decoder_only_loss = torch.tensor(0., device=device)
            residual_pred_norm = torch.tensor(0., device=device)

        if residual_kd_losses:
            residual_kd_loss = torch.stack(residual_kd_losses).mean()
        else:
            residual_kd_loss = torch.tensor(0., device=device)

        rec_loss = torch.tensor(0., device=device)
        return {
            'pred_loss': pred_loss,
            'rec_loss': rec_loss,
            'decoder_only_loss': decoder_only_loss,
            'residual_pred_norm': residual_pred_norm,
            'reskd_loss': residual_kd_loss,
        }

    def _build_force_length_fn(self, target_token_lengths, input_len, motion_end_token_id):
        """Build a step-aware ``prefix_allowed_tokens_fn`` for forced-length AR.

        For each sample i with target token count N_i, the first N_i generation
        steps are restricted to codebook tokens [<motion_id_0> .. <motion_id_{C-1}>]
        and step >= N_i is restricted to the EOM token. Combined with
        ``max_new_tokens = max(N_i)+1`` this guarantees every sample emits
        exactly N_i codebook tokens followed by EOM.
        """
        motion_tok_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
        codebook_ids = list(range(motion_tok_start,
                                  motion_tok_start + self.m_codebook_size))
        eom_only = [motion_end_token_id]
        target_lens = list(target_token_lengths)

        def fn(batch_id, input_ids):
            # input_ids has shape (cur_len,); generation step (0-indexed) is
            # the index of the token about to be sampled.
            step = int(input_ids.shape[0]) - input_len
            N = target_lens[batch_id]
            if step < N:
                return codebook_ids
            return eom_only

        return fn

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = False,
                        bad_words_ids: List[int] = None,
                        restrict_to_motion: bool = False,
                        target_token_lengths: Optional[List[int]] = None):

        # Device
        self.device = self.language_model.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + " \n " for text in texts]

        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        elif self.lm_type == 'dec':
            # Get motion end marker token ID - use this as the EOS for generation
            motion_end_marker = f'<motion_id_{self.m_codebook_size + 1}>'
            motion_end_token_id = self.tokenizer.convert_tokens_to_ids(motion_end_marker)
            original_eos_token_id = self.tokenizer.eos_token_id

            # Forced-length mode: emit exactly target_token_lengths[i] codebook
            # tokens followed by EOM. Overrides restrict_to_motion behaviour.
            use_force_length = (self.force_length and target_token_lengths is not None
                                and len(target_token_lengths) == source_input_ids.shape[0])

            allowed_fn = None
            gen_bad_words = [[original_eos_token_id]] if original_eos_token_id != motion_end_token_id else None
            min_new_tok = 4
            max_new_tok = max_length

            if use_force_length:
                allowed_fn = self._build_force_length_fn(
                    target_token_lengths, source_input_ids.shape[1], motion_end_token_id)
                gen_bad_words = None
                # +1 for the trailing EOM that closes the sequence.
                max_new_tok = int(max(target_token_lengths)) + 1
                min_new_tok = 1
            elif restrict_to_motion:
                motion_tok_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
                allowed_ids = list(range(motion_tok_start,
                                         motion_tok_start + self.m_codebook_size + 3))
                allowed_fn = lambda _batch_id, _input_ids: allowed_ids
                gen_bad_words = None  # redundant when using allowed_fn

            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=motion_end_token_id,  # Stop at motion end marker
                do_sample=do_sample,
                max_new_tokens=max_new_tok,
                min_new_tokens=min_new_tok,  # Prevent premature EOM in first few steps
                bad_words_ids=gen_bad_words,
                prefix_allowed_tokens_fn=allowed_fn,
            )
            self.tokenizer.padding_side = 'left'

        # Only decode the NEWLY GENERATED part, not the input (which contains padding)
        input_len = source_input_ids.shape[1]
        new_outputs = outputs[:, input_len:]
        
        # Use skip_special_tokens=False to preserve motion tokens
        outputs_string = self.tokenizer.batch_decode(new_outputs,
                                                     skip_special_tokens=False)

        outputs_tokens, cleaned_text = self.motion_string_to_token(
            outputs_string)

        return outputs_tokens, cleaned_text

    def generate_direct_with_hidden(self,
                                    texts: List[str],
                                    max_length: int = 256,
                                    num_beams: int = 1,
                                    do_sample: bool = False,
                                    bad_words_ids: List[int] = None,
                                    restrict_to_motion: bool = False,
                                    temperature: float = 1.0,
                                    top_p: float = 1.0,
                                    target_token_lengths: Optional[List[int]] = None):
        """Generate motion tokens and extract LLM hidden states at motion positions.

        Same autoregressive generation as generate_direct, followed by a
        single forward pass to extract last-layer hidden states at motion
        token positions and feed them directly to the residual predictor.
        This makes inference identical to the training PredLoss pipeline:
            hidden_state -> decode_from_latent -> motion

        Returns:
            outputs_tokens:    list of token tensors per sample
            cleaned_text:      list of decoded text strings
            lm_hidden_states:  list of (1, N_mot, hidden_dim) tensors per sample
                               (None when a sample has no valid motion tokens)
            codebook_indices:  list of (1, N_mot) long tensors per sample
        """
        import torch.nn.functional as F

        self.device = self.language_model.device

        if self.lm_type == 'dec':
            texts_input = [text + " \n " for text in texts]
        else:
            texts_input = texts

        source_encoding = self.tokenizer(texts_input,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        # --- Autoregressive generation (same as generate_direct) ---
        if self.lm_type == 'encdec':
            generated_ids = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        elif self.lm_type == 'dec':
            motion_end_marker = f'<motion_id_{self.m_codebook_size + 1}>'
            motion_end_token_id = self.tokenizer.convert_tokens_to_ids(motion_end_marker)
            original_eos_token_id = self.tokenizer.eos_token_id

            use_force_length = (self.force_length and target_token_lengths is not None
                                and len(target_token_lengths) == source_input_ids.shape[0])

            allowed_fn = None
            gen_bad_words = [[original_eos_token_id]] if original_eos_token_id != motion_end_token_id else None
            min_new_tok = 4
            max_new_tok = max_length

            if use_force_length:
                allowed_fn = self._build_force_length_fn(
                    target_token_lengths, source_input_ids.shape[1], motion_end_token_id)
                gen_bad_words = None
                max_new_tok = int(max(target_token_lengths)) + 1
                min_new_tok = 1
            elif restrict_to_motion:
                motion_tok_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
                allowed_ids = list(range(motion_tok_start,
                                         motion_tok_start + self.m_codebook_size + 3))
                allowed_fn = lambda _batch_id, _input_ids: allowed_ids
                gen_bad_words = None

            gen_kwargs = dict(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=motion_end_token_id,
                do_sample=do_sample,
                max_new_tokens=max_new_tok,
                min_new_tokens=min_new_tok,
                bad_words_ids=gen_bad_words,
                prefix_allowed_tokens_fn=allowed_fn,
            )
            if do_sample:
                gen_kwargs['temperature'] = temperature
                if top_p < 1.0:
                    gen_kwargs['top_p'] = top_p
            generated_ids = self.language_model.generate(**gen_kwargs)
            self.tokenizer.padding_side = 'left'

        # --- Forward pass to extract hidden states + project ---
        with torch.no_grad():
            attn_mask = (generated_ids != self.tokenizer.pad_token_id).long()
            fwd_out = self.language_model(
                input_ids=generated_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            last_hidden = fwd_out.hidden_states[-1]  # (B, T, hidden_dim)

        motion_id_start = self.tokenizer.convert_tokens_to_ids('<motion_id_0>')
        motion_id_end = motion_id_start + self.m_codebook_size

        # Causal shift: hidden[t] predicts token[t+1]
        shift_hidden = last_hidden[:, :-1, :]
        shift_ids = generated_ids[:, 1:]
        is_codebook = (shift_ids >= motion_id_start) & (shift_ids < motion_id_end)

        lm_hidden_states = []
        codebook_indices = []
        for i in range(generated_ids.shape[0]):
            mask_i = is_codebook[i]
            n_mot = mask_i.sum().item()
            if n_mot > 0:
                hidden_i = shift_hidden[i, mask_i]  # (N_mot, hidden_dim)
                lm_hidden_states.append(hidden_i.unsqueeze(0))  # (1, N_mot, hidden_dim)
                # Extract codebook indices (token_id - motion_id_start)
                code_idx_i = shift_ids[i, mask_i] - motion_id_start  # (N_mot,)
                codebook_indices.append(code_idx_i.unsqueeze(0))  # (1, N_mot)
            else:
                lm_hidden_states.append(None)
                codebook_indices.append(None)

        # --- Parse tokens (same as generate_direct) ---
        input_len = source_input_ids.shape[1]
        new_outputs = generated_ids[:, input_len:]
        outputs_string = self.tokenizer.batch_decode(new_outputs,
                                                     skip_special_tokens=False)
        outputs_tokens, cleaned_text = self.motion_string_to_token(
            outputs_string)

        return outputs_tokens, cleaned_text, lm_hidden_states, codebook_indices

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None,
                             return_hidden: bool = False,
                             target_token_lengths: Optional[List[int]] = None):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        # Use same format as training: just caption followed by motion tokens
                        # Training format: "caption \n motion_tokens<eos>"
                        # So generation input should be: "caption"
                        tasks = [{
                            'input':
                            ['<Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)

                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
                    
            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(
                    motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append(
                        '>'.join(motion_strings_old[i].split('>')[:split]) +
                        '>')

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(
                    motion_tokens, lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts,
                                                    stage)

            # Determine max_new_tokens based on codebook size
            # Single-scale VQ-VAE: 1 token per frame -> max 128 tokens
            max_new_tokens = 128
            
            # Greedy decoding at eval-time: match Stage 3 Phase A cache (greedy)
            # so the decoder/residual_predictor see the same token distribution
            # at training and test time.
            if return_hidden:
                outputs_tokens, cleaned_text, lm_hidden_states, codebook_indices = \
                    self.generate_direct_with_hidden(inputs,
                                                     max_length=max_new_tokens,
                                                     num_beams=1,
                                                     do_sample=False,
                                                     restrict_to_motion=True,
                                                     target_token_lengths=target_token_lengths)
                return outputs_tokens, lm_hidden_states, codebook_indices
            else:
                outputs_tokens, cleaned_text = self.generate_direct(inputs,
                                                                    max_length=max_new_tokens,
                                                                    num_beams=1,
                                                                    do_sample=False,
                                                                    restrict_to_motion=True,
                                                                    target_token_lengths=target_token_lengths)

                return outputs_tokens

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths)

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                # bad_words_ids=self.bad_words_ids
            )
            
            # Clean up the prompt prefix from generated text (for test/eval/val stage)
            if stage in ['test', 'val']:
                cleaned_text_final = []
                for text in cleaned_text:
                    # The text format is like: "Generate text: <Motion_Placeholder> \n actual_description"
                    # We need to extract only the actual description part
                    
                    # Try to find content after " \n " (the separator used in GPT2 decoder)
                    if ' \n ' in text:
                        text = text.split(' \n ', 1)[-1]
                    
                    # Remove "Generate text: <Motion_Placeholder>" prefix and variants
                    text = text.replace('Generate text: <Motion_Placeholder>', '').strip()
                    text = text.replace('Generate text with', '').strip()
                    text = text.replace('<Motion_Placeholder>', '').strip()
                    
                    # Remove frame placeholder patterns like "196 frames:"
                    import re
                    text = re.sub(r'^\d+\s*frames?:?\s*', '', text)
                    
                    # Remove any leading/trailing whitespace and newlines
                    text = text.strip(' \n\t')
                    
                    # Remove surrounding quotes (both single and double)
                    text = text.strip('"\'')
                    
                    cleaned_text_final.append(text)
                return cleaned_text_final
            
            return cleaned_text

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        """
        Extract motion tokens from generated strings.

        This method handles two cases:
        1. With markers: <motion_id_start>...<motion_id_end> (training format)
        2. Without markers: directly extract all <motion_id_X> where X < codebook_size
        """
        import re
        motion_tokens = []
        output_string = []
        
        for i in range(len(motion_string)):
            start_marker = f'<motion_id_{self.m_codebook_size}>'
            end_marker = f'<motion_id_{self.m_codebook_size + 1}>'
            
            # Try to find content between start and end markers first
            has_start = start_marker in motion_string[i]
            has_end = end_marker in motion_string[i]
            
            if has_start and has_end:
                # Original logic: extract content between markers
                string = self.get_middle_str(
                    motion_string[i], start_marker, end_marker)
                string_list = string.split('><')
                token_list = [
                    int(t.split('_')[-1].replace('>', ''))
                    for t in string_list[1:-1]
                ]
            else:
                # Fallback: extract ALL motion tokens from the string
                # Pattern matches <motion_id_X> where X is a number
                pattern = r'<motion_id_(\d+)>'
                matches = re.findall(pattern, motion_string[i])
                
                # Filter to only include valid motion tokens (0 to codebook_size-1)
                # Exclude start marker (1280), end marker (1281), and mask token (1282)
                token_list = [
                    int(m) for m in matches 
                    if int(m) < self.m_codebook_size
                ]
            
            # Ensure at least one token is returned
            if len(token_list) == 0:
                token_list = [0]
                
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                start_marker, '').replace(end_marker, '').replace(
                '<Motion_Placeholder>', ''))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><motion_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',
            motion_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i]))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
