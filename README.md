<div align="center">

# MUGEN

### A Unified Framework for Efficient Motion Understanding and Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/zy22b/MUGEN)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**No codebook, one draw.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [News](#-news)
- [Setup](#️-setup)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Repository Layout](#-repository-layout)
- [Citation](#️-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

**MUGEN** is a unified motion-language framework: one model turns a description
into human motion and describes an observed motion in words. Both directions
share a single motion representation, and that representation is continuous.

Unified motion-language systems have traditionally coupled the two directions
through a shared discrete motion codebook, but quantization limits generation
quality. The strongest generators buy that quality back at growing cost:
stacked residual codebooks enlarge the representation, while masked decoding
stages, long autoregressive rollouts and denoising chains stretch inference. And
none of that decoding machinery serves understanding. MUGEN pays neither cost.

- 🎯 **Adaptive-length autoencoder (ALAE).** Cross-attention compresses a clip of
  *any* length into **K continuous latent slots**, and a second cross-attention
  stack expands those slots back to any requested frame count. Decoder queries
  encode a frame's *relative phase* in the clip rather than an absolute index,
  which is what lets one decoder serve every length. There is no codebook and no
  quantization anywhere in the pipeline.
- 🔀 **Depth-routed hidden states.** A text-conditioned router gives each latent
  slot its own soft mixture over *all* transformer layers, so a slot reads from
  the depth it needs instead of squeezing every piece of motion evidence through
  the final layer. Both halves of the routing logit are tanh-bounded, so no
  logit margin can saturate the softmax and kill the gradient through the
  text-conditional structure.
- 🎲 **Calibrated latent head.** A rank-r plus diagonal covariance over the whole
  flattened latent set, trained by exact maximum likelihood. One draw therefore
  carries the text-conditional variance that a description permits, *correlated
  across slots*, which is what a single-step sampler has to supply all at once.

Generating a motion costs **K language-model steps, one draw, and one decoder
pass**. No iterative refinement, no residual stages, no denoising chain.

The pipeline is two stages: train the autoencoder (Stage 1), then train the
language model against the frozen autoencoder (Stage 2).

```
            ┌──────────── Stage 1: ALAE ────────────┐
motion ────►│ conv trunk ─► cross-attn ─► K latents │────► cross-attn ─► motion
            └───────────────────────────────────────┘         (frozen in Stage 2)
                                    ▲   │
            ┌───────────────────────┴───┴───────────── Stage 2: LM ──────────────┐
text ──────►│ GPT-2 + <MOT> ─► K rollout steps ─► layer router ─► calibrated head│
            └───────────────────────────────────────────────────────────────────┘
motion ────► K latents ─► motion projector ─► GPT-2 ─► caption      (understanding)
```

---

## 📰 News

| Date | Update |
|------|--------|
| 🎉 **Jul 2026** | Code released, and the K=2 HumanML3D model is on [HuggingFace](https://huggingface.co/zy22b/MUGEN) |

---

## 🛠️ Setup

### Requirements

- Python 3.11
- PyTorch 2.0+
- CUDA 11.8+ for training (inference runs on CPU)

### Installation

```bash
# Clone the repository
git clone https://github.com/JYe16/MUGEN
cd MUGEN

# Create conda environment
conda create -n mugen python=3.11 -y
conda activate mugen

# Install dependencies
pip install -r requirements.txt

# Download the GPT-2 backbone into deps/gpt2
bash prepare/prepare_gpt2.sh
```

> **Note on `requirements.txt`.** It pins the PyTorch nightly CUDA 12.8 index so
> that Blackwell-class GPUs (sm_100) work. On older hardware, drop the
> `--extra-index-url` line and install the stable PyTorch build for your CUDA
> version instead.

### Dataset Preparation

1. **HumanML3D.** Follow [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
   to download and preprocess the dataset, then place it under
   `datasets/humanml3d/`.
2. **SnapMoGen** (optional, for the second benchmark, roughly 16.5 GB). Run
   `python download_snap_dataset.py --save_dir datasets/`, which writes
   `datasets/SnapMoGen/`.
3. **Evaluation encoders.** Run `bash prepare/download_evaluators.sh`. These are
   only needed for metrics and for the Stage-1 perceptual loss; generation and
   captioning do not use them.

<details>
<summary>📁 Expected directory structure</summary>

```
MUGEN/
├── datasets/
│   ├── humanml3d/
│   │   ├── new_joint_vecs/   # motion features, (T, 263) per clip
│   │   ├── new_joints/       # joint positions
│   │   ├── texts/            # captions
│   │   ├── train.txt val.txt test.txt
│   │   └── Mean.npy Std.npy  # feature statistics
│   └── snapmogen/            # optional
├── deps/
│   ├── gpt2/                                        # prepare/prepare_gpt2.sh
│   ├── glove/                                       # word vectorizer
│   ├── t2m/t2m/text_mot_match/model/finest.tar      # HumanML3D evaluator
│   └── snapmogen/evaluator/.../model/               # SnapMoGen evaluator
└── checkpoints/
    ├── alae_k2_dec.pt        # Stage-1 autoencoder
    └── mugen_humanml3d_k2.ckpt   # Stage-2 model
```

The paths above are the ones the configs read. If you keep data elsewhere,
override `DATASET.HUMANML3D.ROOT` and the `deps/` paths in
[`configs/assets.yaml`](configs/assets.yaml).

</details>

---

## 🚀 Quick Start

The fastest path needs no dataset and no local checkpoint. The published model
carries its own tokenizer and its own HumanML3D feature statistics, so it can
hand back motion in real units on its own.

```bash
python demo_hf.py --text "a person walks forward and then waves with the right hand." --length 120
```

Or in a few lines of Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "zy22b/MUGEN", trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained("zy22b/MUGEN")

# text -> motion
features = model.generate_motion(
    ["a person walks forward and then waves with the right hand."],
    lengths=[120], tokenizer=tokenizer,
)                                        # (1, 120, 263)
joints = model.features_to_joints(features)   # (1, 120, 22, 3) in metres

# motion -> text
print(model.generate_caption(features, tokenizer=tokenizer))

# the latent interface both directions share
latents = model.encode_motion(features)       # (1, 2, 512): a whole clip, two vectors
longer = model.decode_motion(latents, 200)    # the same latents at a new length
```

See the [model card](https://huggingface.co/zy22b/MUGEN) for the full inference
tutorial, including batching, temperature, and reproducible draws.

To score the released model on the HumanML3D test set with the project's own
evaluation harness, see [Evaluation](#-evaluation).

---

## 🎓 Training

### Stage 1: the adaptive-length autoencoder

```bash
python alae_train.py \
    --data_root datasets/humanml3d/ \
    --k 2 \
    --latent_dim 512 \
    --hidden_dim 512 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --lambda_percept 10 \
    --lambda_ortho 1 \
    --lambda_latent_decorr 0.25 \
    --num_epochs 500 \
    --batch_size 512
```

| Flag | Default | Notes |
|---|---|---|
| `--k` | 64 | Number of latent slots. The released model uses 2. |
| `--latent_dim` | 512 | Width of one slot. |
| `--lambda_percept` | 10 | Weight of the perceptual loss (needs `deps/t2m/`). |
| `--lambda_ortho` | 1 | Orthogonality regulariser on the latent queries. |
| `--lambda_latent_decorr` | 0 | Latent decorrelation. **Set this.** See below. |
| `--percept_cosine` | off | Add a directional term to the perceptual loss. |
| `--disable_perceptual` | off | Skip the perceptual encoders entirely (ablation). |
| `--eval_t2m_every` | 5 | Run the reconstruction FID evaluation every N epochs. |

> **⚠️ Set `--lambda_latent_decorr`.** Without it, several latent slots converge
> to *identical* targets (centered cosine similarity 1.000 between slots). Stage 2
> then has no opportunity for slot-wise routing to specialise, because the slots
> it is asked to distinguish are copies of one another. The released Stage-1
> model was trained with `0.25`; its checkpoint reports a maximum absolute
> off-diagonal latent correlation of 0.007.

Two checkpoints are written per run into `--working_dir`: on HumanML3D
`best_loss.pt` (lowest validation loss) and `best_tm2t.pt` (lowest
reconstruction FID); on SnapMoGen `snap-alae-k<K>.pt` and
`snap-alae-k<K>_tm2t.pt`. Copy the one you want to
`checkpoints/alae_k<K>_dec.pt`, which is what the Stage-2 HumanML3D config
reads.

The checkpoint is self-contained: `alae_train.py` embeds the per-dimension latent
mean and standard deviation next to the weights. Stage 2 reads those statistics
to standardise the space its head predicts in, so they must travel with the
weights. For a checkpoint that predates this, regenerate them with
`scripts/compute_latent_stats.py` and merge them in with
`scripts/merge_latent_stats_into_ckpt.py`.

### Stage 2: the language model

```bash
# HumanML3D flagship (K=2, layer-routed head, joint generation + understanding)
python llm_train_alae.py --cfg configs/server/alae_k2_nll_whs_r2_dec.yaml

# SnapMoGen flagship (K=4)
python llm_train_alae.py --cfg configs/server/snap_alae_k4_nll_whs_r2.yaml

# Reference: plain autoregressive head, no layer router
python llm_train_alae.py --cfg configs/r6p/alae_k4.yaml
```

| Config | Dataset | K | Head |
|---|---|---|---|
| `configs/server/alae_k2_nll_whs_r2_dec.yaml` | HumanML3D | 2 | Layer router + calibrated factor head |
| `configs/server/snap_alae_k4_nll_whs_r2.yaml` | SnapMoGen | 4 | Layer router + calibrated factor head |
| `configs/r6p/alae_k4.yaml` | HumanML3D | 4 | Plain autoregressive projection |

<details>
<summary>⚙️ Key Stage-2 settings, and why they are what they are</summary>

| Key | Value | Notes |
|---|---|---|
| `k_motion_tokens` | 2 | Must equal the `k` of the Stage-1 checkpoint. |
| `latent_loss_mode` | `nll` | Exact Gaussian likelihood on the latent set. Required by the factor head. |
| `latent_low_rank` | 64 | Covariance rank. Matches the measured effective dimension of the residual. |
| `lambda_latent_mse` | 2 | Weight of the latent likelihood anchor. |
| `num_cross_attn_layers` | 2 | Depth of the router's text encoder. |
| `gumbel_tau_end` | 1.5 | Converged routing temperature; also the temperature used at inference. |
| `gumbel_anneal_epochs` | 100 | Slow anneal, so exploration outlives the epoch range where a static router locks in. |
| `router_static_scale` / `router_delta_scale` | 4.0 / 4.0 | Bounds on the two logit halves. |
| `lambda_route_mi` | 0.05 | Pushes routing to be decisive per caption but different across captions. |
| `freeze_latent_queries` | true | The router's queries are imported from the Stage-1 checkpoint and frozen. |
| `task` | `gnu` | Trains generation and understanding jointly. |
| `lambda_m2t` | 1.0 | Weight of the captioning branch. |
| `m2t_loss_floor` | 2.8 | Floor on the captioning loss. Removing it degrades **both** directions. |
| `variational` | true | The head is a distribution, not a point estimate. |
| `eval_sample_temperature` | 1.0 | Draw one sample per prompt. This is the reported protocol. |
| `METRIC.FID_SAMPLE_TEMP` | 1.0 | Score FID and R-precision from that draw, not from the distribution mean. |

</details>

Long runs belong in a batch job rather than an interactive session; the Stage-2
HumanML3D recipe takes roughly six hours for 500 epochs on one modern
data-center GPU.

---

## 📊 Evaluation

```bash
# HumanML3D test set, 20 replications, honest sampled protocol
python local_eval.py --cfg configs/server/alae_k2_nll_whs_r2_eval.yaml

# SnapMoGen test set
python local_eval.py --cfg configs/server/snap_alae_k4_nll_whs_r2_eval.yaml
```

Point `TEST.CHECKPOINTS` at the Stage-2 checkpoint you want to score. Reported
metrics are FID, R-precision, Diversity and MultiModality for generation, and
BLEU, ROUGE-L, CIDEr, BERTScore and R-precision for captioning.

**Read the protocol before comparing numbers.**

- **Sampled, not mean-decoded.** With `METRIC.FID_SAMPLE_TEMP: 1.0` every
  reported number comes from a single draw of the calibrated conditional
  distribution. Setting it to `0` decodes the distribution mean instead, which is
  a regression protocol and is not comparable to a generative one.
- **Captioning scores only compare within a dataset.** BLEU and CIDEr respond
  very differently to reference count and caption length, and the two benchmarks
  differ on both. Reading them across datasets can invert the conclusion.
- **MultiModality needs a non-zero temperature.** At temperature 0 the pipeline
  is deterministic and MultiModality collapses to 0 by construction.
- **Checkpoint selection is part of the protocol.** `best_fid.ckpt` and
  `best_gnu.ckpt` select on generation alone and on generation plus
  understanding respectively, and the gap between two such snapshots from one run
  can be the same size as the effect you are trying to measure. State which one
  you used.

`scripts/visualize_alae_recon.py` renders Stage-1 reconstructions, and
`scripts/analyze_router_decision.py` with `scripts/plot_router_decision.py`
reports which transformer layer each latent slot ends up reading from.

---

## 📁 Repository Layout

```
alae_train.py                                  Stage 1 training
llm_train_alae.py                              Stage 2 training
local_eval.py                                  Evaluation on either benchmark
demo_hf.py                                     Run the released model from HuggingFace

adaptive_length_auto_encoder/model.py          AdaptiveLengthAutoEncoder (Stage 1)
motGPT/archs/alae_wrapper.py                   Loads Stage 1, exposes the frozen decoder
motGPT/archs/mgpt_lm.py                        Language-model wrapper and tokenizer setup

motGPT/models/alae_motion_gpt.py               Base model: rollout, calibrated head, M2T branch
motGPT/models/alae_motion_gpt_weighted_hs.py   Adds layer routing
motGPT/models/alae_motion_gpt_weighted_hs_r2.py  Residual text-adaptive router (released model)

motGPT/data/                                   HumanML3D and SnapMoGen data modules
motGPT/metrics/                                Generation and captioning metrics
configs/                                       Flagship recipes and their eval counterparts
scripts/                                       Latent statistics, reconstruction views, router analysis
utils/                                         Evaluation encoders and training helpers
```

`motGPT/` keeps its original package name from MotionGPT, which this codebase
descends from.

---

## 🖊️ Citation

The paper is under review. A citation entry will be added here once it appears;
in the meantime please cite the repository and the model card.

```bibtex
@misc{mugen2026,
  title  = {MUGEN: A Unified Framework for Efficient Motion Understanding and Generation},
  note   = {Under review. Code: https://github.com/JYe16/MUGEN,
            model: https://huggingface.co/zy22b/MUGEN},
  year   = {2026}
}
```

---

## 🙏 Acknowledgements

This project builds on the work of:

- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) - motion dataset and evaluation protocol
- [SnapMoGen](https://github.com/snap-research/SnapMoGen) - second benchmark and its evaluator
- [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) - motion-language framework this codebase descends from
- [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) - motion latent representation
- [GeoMotionGPT](https://github.com/JYe16/GeoMotionGPT) - our earlier discrete-codebook system

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

</div>
