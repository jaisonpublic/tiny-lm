# 📖 Glossary — Key Terminology

> **How to use**: This file is designed as an Obsidian-style quick-reference.
> Other documents in this project link here using `[[GLOSSARY#Term]]` syntax.
> In Obsidian, hover over any `[[GLOSSARY#Term]]` link to see a popup preview.
> In VS Code / GitHub, click links to jump to the definition.

---

## Model Architecture Terms

### Transformer
The foundational neural network architecture (Vaswani et al., 2017) used in all modern language models. It processes sequences of tokens in parallel using [[GLOSSARY#Attention]] mechanisms, unlike older RNN/LSTM models that process tokens one at a time. Our project uses **decoder-only** transformers (GPT-style), which generate text left-to-right.

### Attention
The core mechanism that lets the model look at all previous tokens when predicting the next one. Each token creates a **Query** ("what am I looking for?"), **Key** ("what do I contain?"), and **Value** ("what information do I provide?"). The [[GLOSSARY#d_model]] dimension determines the size of these vectors.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

### Multi-Head Attention (MHA)
Instead of one big attention computation, the model performs [[GLOSSARY#n_heads]] parallel attention operations (each called a "head"), each focusing on different patterns (e.g., one head tracks syntax, another tracks semantics). The results are concatenated and projected back.

```
head_dim = d_model / n_heads
```

### Grouped Query Attention (GQA)
An optimization where multiple query heads share a single key-value head. Controlled by [[GLOSSARY#n_kv_heads]]. Reduces memory usage during inference without significant quality loss.

| Configuration | n_heads | n_kv_heads | Name                    |
| ------------- | ------- | ---------- | ----------------------- |
| Full MHA      | 8       | 8          | Multi-Head Attention    |
| GQA           | 8       | 2          | Grouped Query Attention |
| MQA           | 8       | 1          | Multi-Query Attention   |

### Feed-Forward Network (FFN)
A two-layer neural network applied to each token independently after attention. It's where the model stores "knowledge" learned during training. Size is controlled by [[GLOSSARY#d_ff]].

```
FFN(x) = GELU(x·W1 + b1)·W2 + b2
```

### SwiGLU
An improved FFN activation used in Llama-style models. Uses a gating mechanism:

```
SwiGLU(x) = SiLU(x·W_gate) ⊙ (x·W_up)
```

### Layer Normalization (LayerNorm)
Normalizes activations to have zero mean and unit variance across the feature dimension. Stabilizes training. GPT-2 style uses **Pre-Norm** (normalize before attention/FFN).

### RMSNorm
A faster variant of [[GLOSSARY#Layer Normalization (LayerNorm)]] that only normalizes by root-mean-square (no mean subtraction). Used in Llama-style models.

### Positional Encoding
Since transformers process all tokens simultaneously, they need position information. Two main types:
- **Learned Embeddings** (GPT-2): A lookup table mapping position → vector
- **RoPE** ([[GLOSSARY#Rotary Position Embeddings (RoPE)]]): Rotation-based, generalizes better to longer sequences

### Rotary Position Embeddings (RoPE)
Encodes position by rotating query/key vectors. Better than learned embeddings for length generalization. Controlled by [[GLOSSARY#rope_theta]] frequency base.

### Weight Tying
Sharing the same weight matrix between the token embedding layer and the final output (LM head) layer. Saves ~25% of parameters since the vocab embedding is a large matrix (`vocab_size × d_model`).

---

## Model Configuration Parameters

### vocab_size
**What**: Number of unique tokens the model can understand.
**Range**: 4,096 – 32,000 for small models.
**Impact**: Directly affects the embedding layer size (`vocab_size × d_model` parameters). Larger vocab = more tokens learned as single units (efficient) but bigger model.
**Our project**: 8,192 (code-focused) or 16,384 (multilingual).

### d_model
**What**: The "width" of the model — dimension of every hidden representation vector. Also called `n_embd` or `hidden_size`.
**Range**: 128 – 4,096 depending on model size.
**Impact**: Quadratically affects parameter count. Doubling d_model roughly **4×** the parameters. This is the most important size knob.
**Relationship**: Each token is represented as a d_model-dimensional vector at every layer.

### n_layers
**What**: Number of transformer blocks stacked vertically. Also called `n_layer` or `num_hidden_layers`.
**Range**: 2 – 96 (2–12 for our small models).
**Impact**: Linearly increases parameters and compute. More layers = model can learn more complex patterns but trains slower.
**Rule of thumb**: For small models, better to increase [[GLOSSARY#d_model]] first, then add layers.

### n_heads
**What**: Number of parallel attention heads in [[GLOSSARY#Multi-Head Attention (MHA)]].
**Range**: 2 – 64 (must divide d_model evenly).
**Impact**: More heads = more diverse attention patterns. Each head has dimension `d_model / n_heads`.
**Rule of thumb**: `head_dim` of 64–128 works well. E.g., d_model=384 → 6 heads (head_dim=64).

### n_kv_heads
**What**: Number of key-value heads for [[GLOSSARY#Grouped Query Attention (GQA)]]. Only used in Llama-style models.
**Range**: 1 – n_heads.
**Impact**: Lower = less memory during inference. Set to `n_heads` for standard MHA.

### d_ff
**What**: Hidden dimension of the [[GLOSSARY#Feed-Forward Network (FFN)]]. Also called `n_inner` or `intermediate_size`.
**Range**: Usually 4× d_model for GPT-2, or `(8/3) × d_model` for SwiGLU.
**Impact**: Linearly increases parameters. This is where the model stores factual knowledge.

### context_length
**What**: Maximum number of tokens the model can see at once. Also called `n_positions`, `n_ctx`, or `max_position_embeddings`.
**Range**: 128 – 8,192 for small models.
**Impact**: Quadratically increases memory during training (attention matrix is `seq_len × seq_len`). For MicroPython scripts (typically 10–50 lines), 512 tokens is plenty.

### dropout
**What**: Probability of randomly zeroing activations during training. Prevents [[GLOSSARY#Overfitting]].
**Range**: 0.0 – 0.3.
**Impact**: Higher = stronger regularization. Set to 0.0 during inference. Use 0.1 for small datasets, 0.0 for large datasets.

### rope_theta
**What**: Base frequency for [[GLOSSARY#Rotary Position Embeddings (RoPE)]]. Only used in Llama-style models.
**Default**: 10,000.
**Impact**: Higher values allow better extrapolation to longer sequences. 500,000 used in Llama 3.

---

## Training Parameters

### Learning Rate
**What**: Step size for gradient updates. The most critical hyperparameter.
**Range**: 1e-5 – 1e-3.
**Impact**: Too high → training diverges (loss explodes). Too low → training stalls (loss decreases very slowly).
**Rule of thumb**: 3e-4 for pre-training, 5e-5 for fine-tuning.

### Warmup
**What**: Gradually increasing the learning rate from 0 to the target value over the first N steps.
**Range**: 1% – 10% of total training steps.
**Impact**: Prevents early instability when the model hasn't learned anything yet and gradients are noisy.

### Cosine Scheduler
**What**: After warmup, the learning rate follows a cosine curve from peak to [[GLOSSARY#min_lr]].
**Why**: Smoothly decreasing LR lets the model "settle" into a good solution.

```
lr(t) = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × t / T))
```

### min_lr
**What**: The minimum learning rate at the end of the cosine schedule.
**Range**: Usually max_lr / 5 to max_lr / 10.
**Impact**: Too low and training stalls at the end. Too high and the model keeps overshooting.

### Weight Decay
**What**: L2 regularization — penalizes large weights to prevent [[GLOSSARY#Overfitting]].
**Range**: 0.01 – 0.1.
**Impact**: Higher = stronger regularization. Applied to all parameters except biases and LayerNorm.

### Gradient Clipping
**What**: Caps gradient magnitudes to prevent [[GLOSSARY#Exploding Gradients]].
**Value**: Usually 1.0.
**Impact**: Prevents training instability from occasional large gradients.

### Batch Size
**What**: Number of samples processed before one gradient update.
**Formula**: `effective_batch = micro_batch × accumulate_grad_batches × num_gpus`
**Impact**: Larger batch = smoother gradient estimates, but needs more memory. Often compensated with [[GLOSSARY#Gradient Accumulation]].

### Gradient Accumulation
**What**: Simulates large batch sizes by accumulating gradients over multiple micro-batches before updating weights.
**Example**: `batch_size=16, accumulate=4` → effective batch of 64.
**Impact**: Allows training with larger effective batch sizes on limited GPU memory.

### Precision (bf16-mixed)
**What**: Using 16-bit floating point for forward/backward passes while keeping 32-bit master weights.
**Types**: `fp16`, `bf16` (wider range, preferred on RTX 5070), `fp32` (full precision).
**Impact**: ~2× speedup, ~50% less memory vs fp32. bf16 is preferred because it doesn't need loss scaling.

### max_steps
**What**: Total number of gradient update steps (not epochs).
**Formula**: `max_steps ≈ total_tokens / (batch_size × seq_len × accumulate)`
**Impact**: Training too few steps = underfitting. Too many = overfitting (especially on small datasets).

---

## Training Concepts

### Overfitting
When the model memorizes training data instead of learning general patterns. Signs: `train_loss` keeps decreasing but `val_loss` starts increasing. Fix with more data, [[GLOSSARY#dropout]], weight decay, or smaller model.

### Underfitting
When the model hasn't learned enough. Signs: Both `train_loss` and `val_loss` remain high. Fix with more training steps, larger model, or higher learning rate.

### Exploding Gradients
When gradient values grow uncontrollably large, causing loss to spike or become NaN. Fixed by [[GLOSSARY#Gradient Clipping]].

### Perplexity (PPL)
Exponential of the cross-entropy loss: `PPL = e^loss`. Measures how "surprised" the model is by the validation data. Lower is better.

| PPL   | Quality   | Meaning                             |
| ----- | --------- | ----------------------------------- |
| < 5   | Excellent | Model predicts next token very well |
| 5–20  | Good      | Solid generation quality            |
| 20–50 | Fair      | Some coherent output, many errors   |
| > 100 | Poor      | Mostly random output                |

### Loss
Cross-entropy between model's predicted token distribution and the actual next token. The primary metric to minimize during training.

### Tokens per Second
Training throughput metric. On an RTX 5070:

| Model Size | Expected tok/s (bf16) |
| ---------- | --------------------- |
| 5M params  | 400K–600K             |
| 25M params | 100K–200K             |
| 50M params | 50K–100K              |
| 80M params | 30K–60K               |

---

## Tokenizer Terms

### BPE (Byte Pair Encoding)
The tokenization algorithm used. Starts with individual bytes, then iteratively merges the most frequent adjacent pairs. Result: common words become single tokens, rare words are split into subwords.

### Vocabulary
The fixed set of tokens the model knows. Every piece of text is encoded as a sequence of token IDs from this vocabulary.

### Special Tokens
Tokens with special meaning not found in regular text:
- `<bos>` — Beginning of Sequence
- `<eos>` — End of Sequence
- `<pad>` — Padding (fills unused slots)
- `<unk>` — Unknown token (fallback)

---

## Export & Deployment Terms

### GGUF
**G**GPT **G**enerated **U**nified **F**ormat. The file format used by `llama.cpp` and Ollama to store model weights, tokenizer, and metadata in a single portable file.

### Quantization
Reducing the precision of model weights to shrink the file size and speed up inference:

| Type   | Bits | Size vs FP16 | Quality Loss | Use Case             |
| ------ | ---- | ------------ | ------------ | -------------------- |
| FP32   | 32   | 2.0×         | None         | Training only        |
| FP16   | 16   | 1.0× (base)  | Negligible   | GPU inference        |
| Q8_0   | 8    | 0.5×         | Minimal      | CPU server           |
| Q4_K_M | 4    | 0.28×        | Small        | Desktop/laptop       |
| Q2_K   | 2    | 0.15×        | Moderate     | Mobile (last resort) |

### Ollama
A tool for running LLMs locally. It manages model downloading, serving (via API on port 11434), and provides a CLI for chat. Compatible with any GGUF model.

### SafeTensors
A safe, fast file format for storing neural network weights. Used as an intermediate step between PyTorch checkpoints and GGUF conversion.

### HuggingFace Format
The standard model format used by the `transformers` library. Consists of `config.json` + `model.safetensors` + tokenizer files. Required as an intermediate step before GGUF conversion.

---

## Data Terms

### Corpus
The entire collection of text used for training. For our project, this includes MicroPython code, documentation, and instruction pairs.

### Tokenized Data (.bin)
Pre-processed binary files where text has been converted to sequences of integer token IDs. This format is fast to load during training.

### Augmentation
Artificially increasing training data by creating variations (renaming variables, changing pin numbers, rephrasing prompts, etc.). Essential when the raw dataset is small.

### PDF Extraction
Converting PDF textbooks/tutorials into plain text suitable for training. Uses libraries like `PyMuPDF` (fitz) or `pdfplumber` to extract text while preserving code blocks and structure.

---

## Hardware Terms

### VRAM
Video RAM on the GPU. Determines the maximum model size and batch size you can train with.

| GPU          | VRAM  | Max Training Model (bf16) |
| ------------ | ----- | ------------------------- |
| RTX 3060     | 12 GB | ~50M params               |
| RTX 4070     | 12 GB | ~50M params               |
| **RTX 5070** | 12 GB | ~50M params               |
| RTX 4090     | 24 GB | ~150M params              |
| RTX 5090     | 32 GB | ~200M params              |

### TF32 (TensorFloat-32)
A special precision mode on NVIDIA Ampere+ GPUs (including RTX 5070) that uses 19-bit precision for matrix multiplications while maintaining FP32 range. Provides ~3× speedup over pure FP32 with negligible accuracy loss.

### Flash Attention
An optimized attention algorithm that reduces GPU memory usage from O(n²) to O(n) by computing attention in tiles. Automatically used via `F.scaled_dot_product_attention` in PyTorch 2.0+.

### torch.compile()
PyTorch's JIT compiler that fuses operations for faster execution. Adds ~30s compilation time on first run but provides 10–30% speedup.
