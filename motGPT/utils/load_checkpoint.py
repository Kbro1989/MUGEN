import torch
import sys


def is_lora_checkpoint(state_dict):
    """Check if checkpoint was saved with LoRA/PEFT wrapper."""
    for key in state_dict.keys():
        if 'base_model.model' in key or 'lora_' in key:
            return True
    return False


def convert_lora_state_dict_to_merged(state_dict):
    """
    Convert LoRA checkpoint state_dict to merged format.
    This removes the PEFT wrapper prefix and merges LoRA weights into base weights.
    
    LoRA format: 
        lm.language_model.base_model.model.transformer.h.0.attn.c_attn.base_layer.weight
        lm.language_model.base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight
        lm.language_model.base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight
    
    Target format:
        lm.language_model.transformer.h.0.attn.c_attn.weight
    """
    from collections import OrderedDict
    
    # Collect LoRA weights
    lora_A = {}
    lora_B = {}
    base_weights = {}
    other_weights = OrderedDict()
    
    for key, value in state_dict.items():
        if 'lora_A' in key:
            # Extract base key: remove lora_A.default.weight and base_model.model
            base_key = key.replace('.lora_A.default.weight', '.weight')
            base_key = base_key.replace('.base_model.model.', '.')
            lora_A[base_key] = value
        elif 'lora_B' in key:
            base_key = key.replace('.lora_B.default.weight', '.weight')
            base_key = base_key.replace('.base_model.model.', '.')
            lora_B[base_key] = value
        elif '.base_layer.' in key:
            # Base weight for LoRA layers
            base_key = key.replace('.base_layer.', '.')
            base_key = base_key.replace('.base_model.model.', '.')
            base_weights[base_key] = value
        elif '.base_model.model.' in key:
            # Non-LoRA weights that still have PEFT wrapper
            new_key = key.replace('.base_model.model.', '.')
            other_weights[new_key] = value
        else:
            # Other weights without PEFT wrapper
            other_weights[key] = value
    
    # Merge LoRA weights: W_merged = W_base + (B @ A) * scaling
    # Note: Default scaling in PEFT is lora_alpha / r, but since we saved
    # during training, the scaling should already be applied
    merged_state_dict = OrderedDict()
    merged_state_dict.update(other_weights)
    
    # Add base weights (for LoRA layers)
    for key, base_value in base_weights.items():
        if key in lora_A and key in lora_B:
            # Merge: W = W_base + B @ A (assuming scaling is 1 or handled by LoRA config)
            # For inference, we just need base weights since we'll apply LoRA separately
            # Actually, for test.py we should just load and apply LoRA properly
            pass
        merged_state_dict[key] = base_value
    
    # For any remaining LoRA keys, we need the merged weights
    # But since test.py doesn't apply LoRA, we need to merge them
    for key in lora_A:
        if key in lora_B and key in base_weights:
            # W_merged = W_base + alpha/r * B @ A
            # Typical default: alpha=16, r=8 -> scaling = 2
            # But we'll use scaling=1 since the training might have different settings
            A = lora_A[key]  # (r, in_features)
            B = lora_B[key]  # (out_features, r)
            base = base_weights[key]
            # For Conv1D in GPT2, weight is (out_features, in_features)
            # LoRA: B @ A gives (out_features, in_features)
            scaling = 2.0  # Default: lora_alpha(16) / r(8) = 2
            try:
                lora_delta = B @ A * scaling
                merged_state_dict[key] = base + lora_delta
            except Exception as e:
                print(f"Warning: Could not merge LoRA for {key}: {e}")
                merged_state_dict[key] = base
    
    return merged_state_dict


def apply_lora_for_testing(cfg, model, logger=None):
    """Apply LoRA wrapper to model before loading LoRA checkpoint."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        if logger:
            logger.warning("PEFT not installed. Cannot load LoRA checkpoint.")
        return False
    
    # Get LoRA config from cfg - check multiple possible locations
    lora_cfg = None
    if hasattr(cfg, 'LORA') and cfg.LORA is not None:
        lora_cfg = cfg.LORA
    elif hasattr(cfg, 'model') and hasattr(cfg.model, 'params') and hasattr(cfg.model.params, 'LORA'):
        lora_cfg = cfg.model.params.LORA
    
    if lora_cfg is None:
        if logger:
            logger.warning("No LORA config found in cfg.LORA or cfg.model.params.LORA")
        return False
    
    # Check if LORA is enabled
    lora_enabled = getattr(lora_cfg, 'ENABLED', False)
    if not lora_enabled:
        if logger:
            logger.info("LORA.ENABLED is False, skipping LoRA wrapper")
        return False
    
    r = getattr(lora_cfg, 'r', 8)
    lora_alpha = getattr(lora_cfg, 'lora_alpha', 16)
    lora_dropout = getattr(lora_cfg, 'lora_dropout', 0.05)
    target_modules = getattr(lora_cfg, 'target_modules', ['c_attn', 'c_proj'])
    if hasattr(target_modules, '__iter__') and not isinstance(target_modules, str):
        target_modules = list(target_modules)
    train_embeddings = getattr(lora_cfg, 'train_embeddings', 'motion_only')
    
    if logger:
        logger.info(f"Applying LoRA wrapper for testing:")
        logger.info(f"  r={r}, alpha={lora_alpha}, targets={target_modules}")
    
    # Get GPT2 model
    gpt2 = model.lm.language_model
    
    # Determine modules_to_save based on train_embeddings setting
    modules_to_save = None
    if train_embeddings == 'all':
        modules_to_save = ['wte', 'lm_head']
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,  # Set to True for testing
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    
    # Apply LoRA
    model.lm.language_model = get_peft_model(gpt2, peft_config)
    
    if logger:
        logger.info("LoRA wrapper applied successfully")
    
    return True


def load_pretrained(cfg, model, logger=None, phase="train"):
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    
    # Handle nested state_dict (some checkpoints have double-nested structure)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and len(state_dict) == 1:
        if logger is not None:
            logger.info("Detected nested state_dict, unwrapping...")
        state_dict = state_dict["state_dict"]
    
    # Check if this is a LoRA checkpoint
    if is_lora_checkpoint(state_dict):
        # The model may ALREADY carry the peft wrapper: MLM.__init__ calls
        # get_peft_model() whenever its `lora` param is true (see
        # motGPT/archs/mgpt_lm.py). Then the checkpoint key names are by
        # construction identical to the model's own, and rewriting them is not
        # merely unnecessary but destructive: convert_lora_state_dict_to_merged
        # strips the `.base_model.model.` infix, after which NONE of the lm.*
        # keys match and the strict=False load below drops every LM weight in
        # silence (2026-07-24: a 26h Qwen3-1.7B run was evaluated at chance
        # level this way). Only fall through to the wrapper/merge path when the
        # model is genuinely un-wrapped.
        model_is_wrapped = any(
            'base_model.model' in k for k in model.state_dict().keys()
        )
        if model_is_wrapped:
            if logger:
                logger.info(
                    "Detected LoRA checkpoint and the model is already "
                    "peft-wrapped; loading keys as-is (no conversion).")
        else:
            if logger:
                logger.info("Detected LoRA checkpoint, applying LoRA wrapper...")

            # Try to apply LoRA wrapper to model first
            lora_applied = apply_lora_for_testing(cfg, model, logger)

            if not lora_applied:
                # If can't apply LoRA, try to merge weights
                if logger:
                    logger.info("Could not apply LoRA, attempting to merge LoRA weights...")
                state_dict = convert_lora_state_dict_to_merged(state_dict)
    
    # Handle embedding size mismatch (model may have extended vocab for motion tokens)
    model_state = model.state_dict()
    for key in ['lm.language_model.transformer.wte.weight', 'lm.language_model.lm_head.weight']:
        if key in state_dict and key in model_state:
            ckpt_shape = state_dict[key].shape
            model_shape = model_state[key].shape
            if ckpt_shape != model_shape:
                if logger:
                    logger.info(f"Handling size mismatch for {key}: checkpoint {ckpt_shape} -> model {model_shape}")
                # Copy checkpoint weights into model's larger tensor
                new_weight = model_state[key].clone()
                min_size = min(ckpt_shape[0], model_shape[0])
                new_weight[:min_size] = state_dict[key][:min_size]
                state_dict[key] = new_weight

    # Optional Stage-3 safety net:
    # When the active model was trained with a frozen external VAE and a
    # downstream merge step produced the decoder/residual predictor from an
    # independent DVQ checkpoint (cfg.CODEBOOK_PATH), the Lightning checkpoint
    # may still contain stale Stage 2 `vae.*` weights. In that narrow case the
    # caller can set cfg.TEST.PRESERVE_EXTERNAL_VAE=True to drop those keys.
    #
    # IMPORTANT: this must stay OFF by default. Stage 2 configs such as
    # residual_wing.yaml set `unfreeze_decoder_and_predictor: true`, which
    # means the decoder and residual predictor are actively trained and their
    # updated weights live under the `vae.*` prefix in best.ckpt. Dropping
    # those keys at test time silently reverts the VAE to the initial
    # CODEBOOK_PATH weights while the LM stays trained, producing a massive
    # metric regression (FID/R-precision collapse).
    preserve_external_vae = bool(
        getattr(getattr(cfg, 'TEST', object()), 'PRESERVE_EXTERNAL_VAE', False)
    )
    if (
        phase == "test"
        and preserve_external_vae
        and hasattr(model, 'vae')
        and getattr(cfg, 'CODEBOOK_PATH', None)
    ):
        vae_prefixes = ('vae.', 'motion_vae.')
        vae_keys = [key for key in state_dict.keys() if key.startswith(vae_prefixes)]
        if vae_keys:
            if logger is not None:
                logger.info(
                    f"[TEST.PRESERVE_EXTERNAL_VAE=True] Preserving VAE weights from "
                    f"CODEBOOK_PATH={cfg.CODEBOOK_PATH}; skipping {len(vae_keys)} "
                    f"VAE keys from {ckpt_path}"
                )
            state_dict = {
                key: value for key, value in state_dict.items()
                if not key.startswith(vae_prefixes)
            }
    elif phase == "test" and logger is not None:
        vae_key_count = sum(
            1 for k in state_dict.keys() if k.startswith(('vae.', 'motion_vae.'))
        )
        if vae_key_count:
            logger.info(
                f"Loading {vae_key_count} VAE keys from checkpoint {ckpt_path} "
                f"(TEST.PRESERVE_EXTERNAL_VAE is off — trained decoder/residual "
                f"predictor weights will be used)"
            )

    # Report what actually lands. BaseModel overrides load_state_dict (it
    # re-injects the metrics/_losses buffers before delegating) and returns
    # None, so the missing/unexpected diff has to be computed here. strict=False
    # stays for the legacy partial-load flows, but a checkpoint whose lm.*
    # weights ALL miss is never legitimate -- that is an untrained backbone
    # being evaluated, so fail loudly rather than silently.
    model_keys = set(model.state_dict().keys())
    provided = {k for k in state_dict.keys()
                if '_losses' not in k and 'Metrics' not in k}
    unexpected = provided - model_keys
    missing = {k for k in model_keys - provided
               if not k.startswith(('metrics.', '_losses.'))}
    if logger is not None:
        logger.info(
            f"load_state_dict(strict=False): {len(provided & model_keys)} matched, "
            f"{len(missing)} missing, {len(unexpected)} unexpected")
        for k in sorted(unexpected)[:5]:
            logger.info(f"  unexpected key: {k}")
        for k in sorted(missing)[:5]:
            logger.info(f"  missing key:    {k}")

    ckpt_lm_keys = {k for k in provided if k.startswith('lm.')}
    if ckpt_lm_keys and not (ckpt_lm_keys & model_keys):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} carries {len(ckpt_lm_keys)} lm.* weights but "
            f"none match this model's parameter names: every LM weight would be "
            f"dropped and the run would evaluate an untrained backbone. Refusing "
            f"to continue.")

    model.load_state_dict(state_dict, strict=False)
    model.epoch = ckpt.get('epoch', -1)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    ckpt_path = cfg.TRAIN.PRETRAINED_VAE
    if logger is not None:
        logger.info(f"Loading pretrain vae from {ckpt_path}")

    vae_module = model.vae if hasattr(model, 'vae') else model.motion_vae

    # New standard for DVQ: always load through native DVQ loader
    # (expects unified checkpoint containing both VQ-VAE and residual predictor)
    if hasattr(vae_module, '_load_pretrained_weights'):
        if logger is not None:
            logger.info("Using model-native _load_pretrained_weights (unified DVQ checkpoint expected)")
        vae_module._load_pretrained_weights(ckpt_path)
        if hasattr(model, 'on_pretrained_vae_loaded'):
            model.on_pretrained_vae_loaded()
        return model

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # Support both formats:
    # 1) Lightning checkpoint: {'state_dict': ...}
    # 2) Direct VAE checkpoint (e.g., checkpoints/dvq.pt): top-level state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported VAE checkpoint format: {type(ckpt)}")

    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        # if 'skel_embedding' in k: continue
        # if 'final_layer' in k:continue
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v

    if len(vae_dict) == 0:
        module_state = vae_module.state_dict()
        direct_keys = set(state_dict.keys())
        module_keys = set(module_state.keys())
        if direct_keys == module_keys and all(
            state_dict[key].shape == module_state[key].shape for key in module_keys
        ):
            if logger is not None:
                logger.info("Detected bare VAE checkpoint without vae./motion_vae. prefix; loading directly")
            vae_module.load_state_dict(state_dict, strict=True)
            if hasattr(model, 'on_pretrained_vae_loaded'):
                model.on_pretrained_vae_loaded()
            return model

        if logger is not None:
            sample_keys = list(state_dict.keys())[:10]
            logger.warning("No motion_vae./vae. keys found when loading PRETRAINED_VAE")
            logger.warning(f"Checkpoint keys sample: {sample_keys}")
        if hasattr(vae_module, '_load_pretrained_weights'):
            if logger is not None:
                logger.info("Falling back to model-native _load_pretrained_weights")
            vae_module._load_pretrained_weights(ckpt_path)
            return model
        raise KeyError("Could not extract VAE weights from PRETRAINED_VAE checkpoint")

    if hasattr(model, 'vae'):
        model.vae.load_state_dict(vae_dict, strict=True)
    else:
        model.motion_vae.load_state_dict(vae_dict, strict=True)

    if hasattr(model, 'on_pretrained_vae_loaded'):
        model.on_pretrained_vae_loaded()
    
    return model
