"""Convert a tiny-lm GPT-2 checkpoint to HuggingFace GPT-2 format.

This script maps the custom tiny-lm weight names to HuggingFace's
GPT2LMHeadModel format, creates the required config.json and
tokenizer files, and saves everything ready for GGUF conversion.

Usage:
    uv run python scripts/export/convert_to_hf_gpt2.py \
        --checkpoint runs/<run>/checkpoints/last.ckpt \
        --model-config configs/models/gpt2-8k-2l-speed.yaml \
        --tokenizer tokenizers/tinystories-8k/tokenizer.pkl \
        --tokenizer-config configs/tokenizers/tinystories-8k.yaml \
        --output-dir exports/hf-gpt2-speed

Output:
    exports/hf-gpt2-speed/
    ├── config.json                # HuggingFace GPT2Config
    ├── model.safetensors          # Weights in HF naming convention
    ├── tokenizer.json             # HF-compatible tokenizer
    ├── special_tokens_map.json    # Special token mappings
    ├── tokenizer_config.json      # Tokenizer settings
    └── generation_config.json     # Default generation params
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import numpy as np

from tiny_lm.model.config import GPT2Config
from tiny_lm.tokenizer.config import TokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tiny-lm GPT-2 checkpoint to HuggingFace format."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to Lightning .ckpt file."
    )
    parser.add_argument(
        "--model-config", required=True, help="Path to model config YAML."
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer.pkl (tiktoken)."
    )
    parser.add_argument(
        "--tokenizer-config", required=True, help="Path to tokenizer YAML config."
    )
    parser.add_argument(
        "--output-dir",
        default="exports/hf-gpt2",
        help="Output directory for HF model files.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Export dtype (default: float32 for max GGUF compatibility).",
    )
    return parser.parse_args()


def load_tinylm_state_dict(
    checkpoint_path: str, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Load state dict from Lightning checkpoint, strip 'model.' prefix."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    # Strip "model." prefix from Lightning wrapping
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def convert_weights(
    sd: dict[str, torch.Tensor], config: GPT2Config
) -> dict[str, torch.Tensor]:
    """Map tiny-lm weight names to HuggingFace GPT2LMHeadModel names.

    Key differences:
    - HF GPT-2 uses Conv1D (transposed weight) for attn and mlp projections,
      but we export as standard Linear (NOT transposed) with safetensors,
      which HF also supports when loading with `from_pretrained`.
    - HF GPT-2 concatenates Q, K, V into a single c_attn weight.
      We concatenate W_q, W_k, W_v along dim=0.
    - tiny-lm LayerNorm uses gamma/beta; HF uses weight/bias.
    """
    hf_sd: dict[str, torch.Tensor] = {}

    # Token + position embeddings
    hf_sd["transformer.wte.weight"] = sd["token_emb.weight"]
    hf_sd["transformer.wpe.weight"] = sd["pos_emb.pos_emb.weight"]

    n_layers = config.n_layers

    for i in range(n_layers):
        prefix = f"blocks.{i}"
        hf_prefix = f"transformer.h.{i}"

        # --- Attention ---
        # HF GPT-2 concatenates Q, K, V into c_attn (shape: [d_model, 3*d_model])
        # HF Conv1D stores weight as [out_features, in_features] but uses
        # transposed multiplication. For safetensors export, we transpose
        # to match HF's Conv1D convention: [in_features, out_features]
        W_q = sd[f"{prefix}.attn.W_q.weight"]  # [d_model, d_model]
        W_k = sd[f"{prefix}.attn.W_k.weight"]  # [d_model, d_model]
        W_v = sd[f"{prefix}.attn.W_v.weight"]  # [d_model, d_model]

        # Concatenate Q, K, V: [3*d_model, d_model] then transpose for Conv1D
        c_attn_weight = torch.cat([W_q, W_k, W_v], dim=0)  # [3*d_model, d_model]
        hf_sd[f"{hf_prefix}.attn.c_attn.weight"] = c_attn_weight.t().contiguous()  # [d_model, 3*d_model]

        # Bias for c_attn: HF expects it, tiny-lm has qkv_bias=False, so create zeros
        hf_sd[f"{hf_prefix}.attn.c_attn.bias"] = torch.zeros(3 * config.d_model)

        # Output projection: transpose for Conv1D
        hf_sd[f"{hf_prefix}.attn.c_proj.weight"] = sd[f"{prefix}.attn.out_proj.weight"].t().contiguous()
        hf_sd[f"{hf_prefix}.attn.c_proj.bias"] = sd[f"{prefix}.attn.out_proj.bias"]

        # --- Layer Norms ---
        hf_sd[f"{hf_prefix}.ln_1.weight"] = sd[f"{prefix}.norm1.gamma"]
        hf_sd[f"{hf_prefix}.ln_1.bias"] = sd[f"{prefix}.norm1.beta"]
        hf_sd[f"{hf_prefix}.ln_2.weight"] = sd[f"{prefix}.norm2.gamma"]
        hf_sd[f"{hf_prefix}.ln_2.bias"] = sd[f"{prefix}.norm2.beta"]

        # --- MLP / Feed-forward ---
        # HF GPT-2 MLP: c_fc (up projection) then c_proj (down projection)
        # transpose for Conv1D convention
        hf_sd[f"{hf_prefix}.mlp.c_fc.weight"] = sd[f"{prefix}.ffn.fc1.weight"].t().contiguous()
        hf_sd[f"{hf_prefix}.mlp.c_fc.bias"] = sd[f"{prefix}.ffn.fc1.bias"]
        hf_sd[f"{hf_prefix}.mlp.c_proj.weight"] = sd[f"{prefix}.ffn.fc2.weight"].t().contiguous()
        hf_sd[f"{hf_prefix}.mlp.c_proj.bias"] = sd[f"{prefix}.ffn.fc2.bias"]

    # Final layer norm
    hf_sd["transformer.ln_f.weight"] = sd["ln_f.gamma"]
    hf_sd["transformer.ln_f.bias"] = sd["ln_f.beta"]

    # LM head: weight-tied with wte. HF handles this via
    # tie_word_embeddings=True in config.json, so we do NOT include
    # lm_head.weight separately (safetensors doesn't allow shared tensors).

    return hf_sd


def create_hf_config(config: GPT2Config, bos_id: int, eos_id: int) -> dict:
    """Create HuggingFace GPT2Config as a dictionary."""
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": config.vocab_size,
        "n_embd": config.d_model,
        "n_layer": config.n_layers,
        "n_head": config.n_heads,
        "n_inner": config.d_ff,
        "n_positions": config.context_length,
        "n_ctx": config.context_length,
        "activation_function": "gelu",  # tiny-lm uses custom GELU (tanh approx)
        "layer_norm_epsilon": 1e-5,
        "resid_pdrop": 0.0,  # Set to 0 for inference
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.45.0",
    }


def create_tokenizer_files(
    tokenizer,
    tok_config: TokenizerConfig,
    output_dir: Path,
) -> None:
    """Create HuggingFace-compatible tokenizer files.

    Since tiny-lm uses tiktoken (BPE), we create a minimal tokenizer.json
    that HF Transformers and llama.cpp can understand.
    """
    bos_token = tok_config.special_tokens.get("bos", "<bos>")
    eos_token = tok_config.special_tokens.get("eos", "<eos>")
    pad_token = tok_config.special_tokens.get("pad", "<pad>")
    unk_token = tok_config.special_tokens.get("unk", "<unk>")

    bos_id = tokenizer.encode_single_token(bos_token)
    eos_id = tokenizer.encode_single_token(eos_token)
    pad_id = tokenizer.encode_single_token(pad_token)
    unk_id = tokenizer.encode_single_token(unk_token)

    # Build the vocab from tiktoken's internal state
    # tiktoken stores mergeable_ranks as bytes -> rank
    vocab = {}
    merge_list = []

    # Access the internal mergeable ranks
    ranks = tokenizer._mergeable_ranks
    # Sort by rank (which IS the token id for base vocab)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])

    for token_bytes, rank in sorted_ranks:
        try:
            token_str = token_bytes.decode("utf-8", errors="replace")
        except Exception:
            token_str = "".join(f"<0x{b:02X}>" for b in token_bytes)
        vocab[token_str] = rank

    # Add special tokens
    vocab[pad_token] = pad_id
    vocab[eos_token] = eos_id
    vocab[bos_token] = bos_id
    vocab[unk_token] = unk_id

    # Create HF tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merge_list,  # We don't have easy access to merges from tiktoken
        },
        "added_tokens": [
            {
                "id": pad_id,
                "content": pad_token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": eos_id,
                "content": eos_token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": bos_id,
                "content": bos_token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": unk_id,
                "content": unk_token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
    }

    # special_tokens_map.json
    special_tokens_map = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
    }

    # tokenizer_config.json
    tokenizer_config = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
        "model_max_length": 256,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }

    # generation_config.json
    generation_config = {
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "max_new_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.95,
        "do_sample": True,
    }

    # Write all files
    _write_json(output_dir / "tokenizer.json", tokenizer_json)
    _write_json(output_dir / "special_tokens_map.json", special_tokens_map)
    _write_json(output_dir / "tokenizer_config.json", tokenizer_config)
    _write_json(output_dir / "generation_config.json", generation_config)

    return bos_id, eos_id


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Converting tiny-lm checkpoint to HuggingFace format...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Model config: {args.model_config}")
    print(f"   Output: {output_dir}")
    print()

    # Load model config
    config = GPT2Config.from_yaml(args.model_config)
    print(f"📐 Model: {config.n_layers}L, {config.d_model}d, {config.n_heads}H, "
          f"vocab={config.vocab_size}, ctx={config.context_length}")

    # Load tokenizer
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    tok_config = TokenizerConfig.from_yaml(args.tokenizer_config)

    # Create tokenizer files and get special token IDs
    bos_id, eos_id = create_tokenizer_files(tokenizer, tok_config, output_dir)
    print(f"🔤 Tokenizer: vocab_size={tokenizer.n_vocab}, bos={bos_id}, eos={eos_id}")

    # Load and convert weights
    print(f"⚙️  Loading checkpoint...")
    sd = load_tinylm_state_dict(args.checkpoint)
    hf_sd = convert_weights(sd, config)

    # Cast dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]
    hf_sd = {k: v.to(target_dtype) for k, v in hf_sd.items()}

    # Save weights as safetensors
    from safetensors.torch import save_file
    safetensors_path = output_dir / "model.safetensors"
    save_file(hf_sd, str(safetensors_path))
    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"💾 Saved weights: {safetensors_path} ({size_mb:.1f} MB)")

    # Save HF config.json
    hf_config = create_hf_config(config, bos_id, eos_id)
    hf_config["torch_dtype"] = args.dtype
    _write_json(output_dir / "config.json", hf_config)
    print(f"📄 Saved config.json")

    # Summary
    n_params = sum(v.numel() for v in hf_sd.values()) / 1e6
    print()
    print(f"✅ Conversion complete!")
    print(f"   Parameters: {n_params:.1f}M")
    print(f"   Dtype: {args.dtype}")
    print(f"   Output: {output_dir}")
    print()
    print(f"📋 Next steps:")
    print(f"   1. Convert to GGUF:")
    print(f"      python llama.cpp/convert_hf_to_gguf.py {output_dir} --outtype f16")
    print(f"   2. Create Ollama model:")
    print(f"      ollama create tiny-stories -f Modelfile")


if __name__ == "__main__":
    main()
