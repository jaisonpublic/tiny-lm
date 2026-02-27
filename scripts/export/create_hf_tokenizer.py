"""Create a proper HuggingFace GPT-2 tokenizer.json from a tiktoken tokenizer.

This creates a tokenizer.json that llama.cpp's convert_hf_to_gguf.py
can understand, with proper BPE merges and pre-tokenizer configuration.
"""

import json
import pickle
import re
import sys
from pathlib import Path

import yaml


def main():
    tokenizer_pkl = sys.argv[1]  # tokenizers/tinystories-8k/tokenizer.pkl
    tokenizer_config_yaml = sys.argv[2]  # configs/tokenizers/tinystories-8k.yaml
    output_dir = sys.argv[3]  # exports/hf-gpt2-speed

    # Load tiktoken tokenizer
    with open(tokenizer_pkl, "rb") as f:
        tokenizer = pickle.load(f)

    with open(tokenizer_config_yaml) as f:
        tok_config = yaml.safe_load(f)

    vocab_size = tokenizer.n_vocab
    special_tokens = tok_config.get("special_tokens", {})

    # Extract mergeable ranks (token_bytes -> rank/id)
    ranks = tokenizer._mergeable_ranks
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])

    # Build vocab: map token_string -> token_id
    # For GPT-2 BPE, we need byte-level encoding
    # GPT-2 uses a byte-to-unicode mapping
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    vocab = {}
    for token_bytes, rank in sorted_ranks:
        # Convert bytes to GPT-2's unicode representation
        token_str = "".join(byte_encoder[b] for b in token_bytes)
        vocab[token_str] = rank

    # Add special tokens at their correct positions
    for name, token_str in special_tokens.items():
        token_id = tokenizer.encode_single_token(token_str)
        vocab[token_str] = token_id

    # Build merges list from the vocabulary
    # In BPE, merges are pairs of tokens that were merged during training
    # We reconstruct them from the vocabulary ordering
    merges = build_merges_from_vocab(vocab, sorted_ranks, byte_encoder)

    # Determine special token info
    bos_token = special_tokens.get("bos", "<bos>")
    eos_token = special_tokens.get("eos", "<eos>")
    pad_token = special_tokens.get("pad", "<pad>")
    unk_token = special_tokens.get("unk", "<unk>")

    bos_id = tokenizer.encode_single_token(bos_token)
    eos_id = tokenizer.encode_single_token(eos_token)
    pad_id = tokenizer.encode_single_token(pad_token)
    unk_id = tokenizer.encode_single_token(unk_token)

    # Build added_tokens list
    added_tokens = []
    for name, token_str in special_tokens.items():
        token_id = tokenizer.encode_single_token(token_str)
        added_tokens.append({
            "id": token_id,
            "content": token_str,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })

    # Create the tokenizer.json in HF format
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    # Write files
    out = Path(output_dir)

    with open(out / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)

    # tokenizer_config.json
    tokenizer_config = {
        "model_type": "gpt2",
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
        "model_max_length": 256,
        "tokenizer_class": "GPT2Tokenizer",
    }
    with open(out / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # special_tokens_map.json
    with open(out / "special_tokens_map.json", "w") as f:
        json.dump({
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
            "pad_token": pad_token,
        }, f, indent=2)

    print(f"✅ Tokenizer files created in {output_dir}")
    print(f"   Vocab size: {len(vocab)}")
    print(f"   Merges: {len(merges)}")
    print(f"   Special tokens: {special_tokens}")


def bytes_to_unicode():
    """GPT-2 byte-to-unicode mapping.
    
    Returns a mapping from byte values to unicode characters.
    This is the standard GPT-2 byte encoding scheme.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def build_merges_from_vocab(vocab, sorted_ranks, byte_encoder):
    """Build BPE merge list from the vocabulary.
    
    For each multi-byte token in the vocab, we find the best split point
    that creates a valid merge (both halves must also be in the vocab with
    lower ranks). This reconstructs the BPE merge order.
    """
    merges = []
    # Single-byte tokens are the base vocabulary (no merges needed)
    # Multi-byte tokens were created by merging two existing tokens
    
    # Build a reverse lookup: id -> encoded string
    id_to_str = {}
    for token_str, token_id in vocab.items():
        if token_id < 256:  # Base byte tokens
            id_to_str[token_id] = token_str
    
    for token_bytes, rank in sorted_ranks:
        if len(token_bytes) <= 1:
            continue  # Single bytes are base vocab
        
        encoded = "".join(byte_encoder[b] for b in token_bytes)
        
        # Try all split points and find valid ones
        best_split = None
        best_max_rank = float("inf")
        
        for i in range(1, len(token_bytes)):
            left_bytes = token_bytes[:i]
            right_bytes = token_bytes[i:]
            
            left_str = "".join(byte_encoder[b] for b in left_bytes)
            right_str = "".join(byte_encoder[b] for b in right_bytes)
            
            if left_str in vocab and right_str in vocab:
                left_rank = vocab[left_str]
                right_rank = vocab[right_str]
                
                # Both parts must have lower rank (been created earlier)
                if left_rank < rank and right_rank < rank:
                    max_rank = max(left_rank, right_rank)
                    if max_rank < best_max_rank:
                        best_max_rank = max_rank
                        best_split = (left_str, right_str)
        
        if best_split:
            merges.append(f"{best_split[0]} {best_split[1]}")
    
    return merges


if __name__ == "__main__":
    main()
