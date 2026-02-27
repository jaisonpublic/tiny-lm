"""Convert tiny-lm model directly to GGUF format.

This bypasses the HF converter and creates a GGUF file directly,
which is simpler for custom tokenizers.
"""

import argparse
import pickle
import struct
from pathlib import Path

import gguf
import numpy as np
import torch

from tiny_lm.model.config import GPT2Config
from tiny_lm.tokenizer.config import TokenizerConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--output", required=True, help="Output .gguf file")
    parser.add_argument("--ftype", default="f16", choices=["f32", "f16"])
    return parser.parse_args()


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    # Strip "model." prefix
    return {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}


def main():
    args = parse_args()
    
    print(f"📦 Converting to GGUF: {args.output}")
    
    # Load config
    config = GPT2Config.from_yaml(args.model_config)
    tok_config = TokenizerConfig.from_yaml(args.tokenizer_config)
    
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    
    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(args.output, "gpt2")
    
    # Set architecture
    gguf_writer.add_architecture()
    gguf_writer.add_block_count(config.n_layers)
    gguf_writer.add_context_length(config.context_length)
    gguf_writer.add_embedding_length(config.d_model)
    gguf_writer.add_feed_forward_length(config.d_ff)
    gguf_writer.add_head_count(config.n_heads)
    gguf_writer.add_layer_norm_eps(1e-5)
    
    # Tokenizer metadata
    gguf_writer.add_tokenizer_model("gpt2")
    
    # Extract vocab from tiktoken
    tokens = []
    scores = []
    
    # Get mergeable ranks (base vocabulary)
    ranks = tokenizer._mergeable_ranks
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
    
    for token_bytes, rank in sorted_ranks:
        try:
            token_str = token_bytes.decode("utf-8", errors="replace")
        except:
            token_str = ""
        tokens.append(token_str)
        scores.append(0.0)  # BPE doesn't use scores
    
    # Add special tokens
    special = tok_config.special_tokens
    bos_id = tokenizer.encode_single_token(special["bos"])
    eos_id = tokenizer.encode_single_token(special["eos"])
    pad_id = tokenizer.encode_single_token(special["pad"])
    
    # Ensure we have vocab_size tokens
    while len(tokens) < config.vocab_size:
        tokens.append(f"<pad_{len(tokens)}>")
        scores.append(0.0)
    
    gguf_writer.add_token_list(tokens[:config.vocab_size])
    gguf_writer.add_token_scores(scores[:config.vocab_size])
    
    # Special token IDs
    gguf_writer.add_bos_token_id(bos_id)
    gguf_writer.add_eos_token_id(eos_id)
    gguf_writer.add_pad_token_id(pad_id)
    
    # Load weights
    sd = load_checkpoint(args.checkpoint)
    
    # Determine dtype
    ftype = gguf.GGMLQuantizationType.F16 if args.ftype == "f16" else gguf.GGMLQuantizationType.F32
    
    # Convert and add tensors
    def add_tensor(name: str, tensor: torch.Tensor):
        data = tensor.detach().cpu().numpy()
        if args.ftype == "f16" and data.dtype == np.float32:
            data = data.astype(np.float16)
        gguf_writer.add_tensor(name, data)
    
    # Token embeddings
    add_tensor("token_embd.weight", sd["token_emb.weight"])
    
    # Position embeddings (GPT-2 specific)
    add_tensor("token_embd.position.weight", sd["pos_emb.pos_emb.weight"])
    
    # Transformer blocks
    for i in range(config.n_layers):
        prefix = f"blocks.{i}"
        blk = f"blk.{i}"
        
        # Attention
        W_q = sd[f"{prefix}.attn.W_q.weight"]
        W_k = sd[f"{prefix}.attn.W_k.weight"]
        W_v = sd[f"{prefix}.attn.W_v.weight"]
        
        # Concatenate QKV
        qkv = torch.cat([W_q, W_k, W_v], dim=0)
        add_tensor(f"{blk}.attn_qkv.weight", qkv)
        
        # QKV bias (zeros since we don't use bias)
        qkv_bias = torch.zeros(3 * config.d_model)
        add_tensor(f"{blk}.attn_qkv.bias", qkv_bias)
        
        # Output projection
        add_tensor(f"{blk}.attn_output.weight", sd[f"{prefix}.attn.out_proj.weight"])
        add_tensor(f"{blk}.attn_output.bias", sd[f"{prefix}.attn.out_proj.bias"])
        
        # Layer norms
        add_tensor(f"{blk}.attn_norm.weight", sd[f"{prefix}.norm1.gamma"])
        add_tensor(f"{blk}.attn_norm.bias", sd[f"{prefix}.norm1.beta"])
        add_tensor(f"{blk}.ffn_norm.weight", sd[f"{prefix}.norm2.gamma"])
        add_tensor(f"{blk}.ffn_norm.bias", sd[f"{prefix}.norm2.beta"])
        
        # FFN
        add_tensor(f"{blk}.ffn_up.weight", sd[f"{prefix}.ffn.fc1.weight"])
        add_tensor(f"{blk}.ffn_up.bias", sd[f"{prefix}.ffn.fc1.bias"])
        add_tensor(f"{blk}.ffn_down.weight", sd[f"{prefix}.ffn.fc2.weight"])
        add_tensor(f"{blk}.ffn_down.bias", sd[f"{prefix}.ffn.fc2.bias"])
    
    # Final layer norm
    add_tensor("output_norm.weight", sd["ln_f.gamma"])
    add_tensor("output_norm.bias", sd["ln_f.beta"])
    
    # Output (tied with token_emb)
    add_tensor("output.weight", sd["token_emb.weight"])
    
    # Write file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    n_params = sum(v.numel() for v in sd.values()) / 1e6
    
    print(f"✅ Created GGUF: {args.output}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Params: {n_params:.1f}M")
    print(f"   Ftype: {args.ftype}")


if __name__ == "__main__":
    main()
