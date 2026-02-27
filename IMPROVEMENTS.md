# 🚀 RTX 5070 GPU Optimization Improvements

> **Date:** 2026-02-14
> **Target Hardware:** NVIDIA GeForce RTX 5070 (Blackwell, Compute Capability 12.0, 12 GB GDDR7, 192 5th-gen Tensor Cores)
>
> This document catalogues optimizations applied to the `tiny-lm` project to fully leverage the RTX 5070 GPU during training and inference.

---

## Table of Contents

1. [Baseline Assessment](#1-baseline-assessment)
2. [Improvement 1 — Flash Attention via SDPA](#2-improvement-1--flash-attention-via-sdpa)
3. [Improvement 2 — `torch.compile()` JIT Compilation](#3-improvement-2--torchcompile-jit-compilation)
4. [Improvement 3 — TF32 Matmul Precision](#4-improvement-3--tf32-matmul-precision)
5. [Improvement 4 — Persistent DataLoader Workers](#5-improvement-4--persistent-dataloader-workers)
6. [Improvement 5 — DataLoader Prefetch Factor](#6-improvement-5--dataloader-prefetch-factor)
7. [Summary of Changes](#7-summary-of-changes)

---

## 1. Baseline Assessment

### What was already working well

| Feature                  | Status | Details                                                              |
| ------------------------ | ------ | -------------------------------------------------------------------- |
| GPU auto-detection       | ✅      | `accelerator="auto"`, `devices="auto"` in Lightning Trainer          |
| bf16-mixed precision     | ✅      | Leverages Tensor Cores via `precision: "bf16-mixed"` in training cfg |
| CUDA 12.8 PyTorch wheels | ✅      | `pyproject.toml` pulls from `pytorch-cu128` index                    |
| Pinned memory            | ✅      | `pin_memory: true` in data config for fast CPU→GPU DMA transfers     |
| GPU monitoring           | ✅      | `GpuStatsMonitor` callback logs memory + temperature via nvidia-smi  |
| Gradient accumulation    | ✅      | 4× accumulation to simulate large effective batch size               |

### What was missing

| #   | Optimization                  | Estimated Speedup         | Priority       |
| --- | ----------------------------- | ------------------------- | -------------- |
| 1   | Flash Attention (SDPA)        | **2–4×**                  | 🔴 Critical     |
| 2   | `torch.compile()`             | **1.3–2×**                | 🔴 Critical     |
| 3   | TF32 matmul precision         | **1.1–1.3×** (FP32 paths) | 🟡 Medium       |
| 4   | Persistent DataLoader workers | Reduces stalls            | 🟡 Medium       |
| 5   | DataLoader prefetch factor    | Smooths pipeline          | 🟢 Nice-to-have |

---

## 2. Improvement 1 — Flash Attention via SDPA

### Problem

The attention module (`tiny_lm/model/attention/multi_head.py`) used a **manual attention implementation** that materializes the full `(batch, heads, seq_len, seq_len)` attention score matrix in VRAM:

```python
# BEFORE — O(n²) memory, no hardware attention acceleration
attn_scores = Q @ K.transpose(-2, -1)
attn_scores = attn_scores / (self.head_dim ** 0.5)
mask = self.causal_mask[:seq_len, :seq_len]
attn_scores = attn_scores.masked_fill(mask, float("-inf"))
attn_weights = torch.softmax(attn_scores, dim=-1)
attn_weights = self.dropout(attn_weights)
context = attn_weights @ V
```

This doesn't leverage **Flash Attention**, which is automatically dispatched by PyTorch's `F.scaled_dot_product_attention()` on modern GPUs.

### Solution

Replace manual attention with PyTorch's native SDPA:

```python
# AFTER — O(n) memory, automatic Flash Attention dispatch
context = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=True,
)
```

### Impact

- **Memory**: O(n²) → O(n) for the attention matrix — allows larger batch sizes or longer sequences
- **Speed**: 2–4× faster attention computation via fused CUDA kernels
- **Compatibility**: `F.scaled_dot_product_attention` is available since PyTorch 2.0 and automatically selects the best backend (FlashAttention, Memory-Efficient, or Math fallback)

### Files Changed

- `tiny_lm/model/attention/multi_head.py`

---

## 3. Improvement 2 — `torch.compile()` JIT Compilation

### Problem

The model ran in PyTorch's default **eager mode**, where each operation is dispatched individually to the GPU. This incurs Python overhead and prevents cross-operation kernel fusion.

### Solution

Wrap the model with `torch.compile()` after construction:

```python
# In scripts/training/train_gpt2.py
model = build_model(model_config)
model = torch.compile(model)  # Triton JIT compilation
```

And similarly in the inference script:

```python
# In scripts/inference/generate_from_ckpt.py
model.to(args.device)
model = torch.compile(model)
model.eval()
```

### Impact

- **Speed**: 1.3–2× overall training speedup through operation fusion
- **How it works**: The Triton compiler fuses sequences like `LayerNorm → Linear → GELU` into single GPU kernels, reducing memory bandwidth pressure and kernel launch overhead
- **First-step cost**: Initial compilation takes ~30–60 seconds; subsequent steps run at full speed

### Files Changed

- `scripts/training/train_gpt2.py`
- `scripts/inference/generate_from_ckpt.py`

---

## 4. Improvement 3 — TF32 Matmul Precision

### Problem

PyTorch defaults to full FP32 precision for float32 matrix multiplications. The RTX 5070's Tensor Cores support **TF32** (TensorFloat-32), which provides near-FP32 accuracy at significantly higher throughput. This was not enabled.

### Solution

Set the float32 matmul precision at the start of training and inference:

```python
torch.set_float32_matmul_precision("high")  # Enable TF32 on Tensor Cores
```

### Impact

- **Speed**: 1.1–1.3× faster for any remaining FP32 operations (even in bf16-mixed mode, some ops stay in FP32)
- **Accuracy**: Negligible difference — TF32 has the same dynamic range as FP32
- **Note**: This complements `bf16-mixed` by accelerating the FP32 master weights and gradient accumulation steps

### Files Changed

- `scripts/training/train_gpt2.py`
- `scripts/inference/generate_from_ckpt.py`

---

## 5. Improvement 4 — Persistent DataLoader Workers

### Problem

With `num_workers: 4`, the DataLoader spawns worker processes that are **destroyed and recreated at every epoch boundary**. This causes periodic stalls as workers reinitialize and re-open memory-mapped `.bin` files.

### Solution

Enable `persistent_workers=True` in both train and validation DataLoaders:

```python
DataLoader(
    ...,
    num_workers=self.num_workers,
    persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
)
```

### Impact

- **Latency**: Eliminates multi-second stalls at epoch boundaries
- **Memory**: Slightly higher baseline RAM usage (workers stay resident)
- **Guard**: Only enabled when `num_workers > 0` (not needed for single-process loading)

### Files Changed

- `tiny_lm/data/bin/data_module.py`

---

## 6. Improvement 5 — DataLoader Prefetch Factor

### Problem

The default `prefetch_factor=2` means each worker only pre-loads 2 batches. On the RTX 5070's PCIe 5.0 x16 interface, the GPU can consume batches faster than they arrive, causing occasional pipeline bubbles.

### Solution

Increase `prefetch_factor` to keep more batches queued:

```python
DataLoader(
    ...,
    prefetch_factor=4 if self.num_workers > 0 else None,
)
```

### Impact

- **Throughput**: Smoother GPU utilization by keeping the data pipeline ahead of the training loop
- **Memory**: Slightly more CPU RAM used for pre-loaded batches (negligible for token data)

### Files Changed

- `tiny_lm/data/bin/data_module.py`

---

## 7. Summary of Changes

| File                                      | Change                                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| `tiny_lm/model/attention/multi_head.py`   | Replaced manual attention with `F.scaled_dot_product_attention` (SDPA / Flash Attention) |
| `scripts/training/train_gpt2.py`          | Added `torch.compile()` and `torch.set_float32_matmul_precision("high")`                 |
| `scripts/inference/generate_from_ckpt.py` | Added `torch.compile()` and `torch.set_float32_matmul_precision("high")`                 |
| `tiny_lm/data/bin/data_module.py`         | Added `persistent_workers=True` and `prefetch_factor=4` to DataLoaders                   |

### Combined Estimated Impact

| Metric            | Before            | After (estimated)      |
| ----------------- | ----------------- | ---------------------- |
| Attention speed   | 1×                | 2–4× (Flash Attention) |
| Overall training  | 1×                | ~3–5× (all combined)   |
| GPU VRAM headroom | Baseline          | +30–50% freed (SDPA)   |
| Data pipeline     | Occasional stalls | Smooth prefetching     |
