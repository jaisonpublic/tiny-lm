"""Chainlit chat app for testing tiny-lm models interactively.

Launch with:
    uv run chainlit run scripts/demo/chainlit_app.py -w

Features:
    - Chat-style interface to generate text from trained models
    - Settings panel to adjust temperature, top-p, max tokens
    - Model selector to switch between available checkpoints
    - Streaming token-by-token output
"""

from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import chainlit as cl
import torch
import yaml

from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.config import GPT2Config
from tiny_lm.tokenizer.config import TokenizerConfig

# ---------------------------------------------------------------------------
# Discovery: find available runs with checkpoints
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizers" / "tinystories-8k"


def discover_runs() -> dict[str, dict]:
    """Scan the runs/ directory and return available model runs."""
    runs: dict[str, dict] = {}
    if not RUNS_DIR.exists():
        return runs

    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        last_ckpt = ckpt_dir / "last.ckpt"
        if not last_ckpt.exists():
            continue

        # Find the model config in the run's saved configs
        configs_dir = run_dir / "configs"
        model_config_path = None
        if configs_dir.exists():
            for f in configs_dir.iterdir():
                if f.name.startswith("gpt2") and "training" not in f.name.lower():
                    # Model configs start with gpt2 but are not training configs
                    with open(f) as fh:
                        contents = yaml.safe_load(fh)
                    if "d_model" in contents:
                        model_config_path = f
                        break

        if model_config_path is None:
            # Try common config locations
            for pattern in ["configs/models/gpt2*.yaml"]:
                candidates = list(configs_dir.glob("gpt2*")) if configs_dir.exists() else []
                if candidates:
                    model_config_path = candidates[0]
                    break

        if model_config_path:
            runs[run_dir.name] = {
                "checkpoint": str(last_ckpt),
                "model_config": str(model_config_path),
                "run_dir": str(run_dir),
            }

    return runs


def load_model(
    checkpoint_path: str,
    model_config_path: str,
    device: str = "cuda",
) -> tuple[GPT2, GPT2Config]:
    """Load a model from a checkpoint and config."""
    config = GPT2Config.from_yaml(model_config_path)
    model = GPT2(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        context_length=config.context_length,
        emb_dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, config


def load_tokenizer():
    """Load the tiktoken tokenizer and its config."""
    tok_pkl = TOKENIZER_DIR / "tokenizer.pkl"
    tok_config_path = PROJECT_ROOT / "configs" / "tokenizers" / "tinystories-8k.yaml"

    with open(tok_pkl, "rb") as f:
        tokenizer = pickle.load(f)

    tok_config = TokenizerConfig.from_yaml(str(tok_config_path))
    bos_id = tokenizer.encode_single_token(tok_config.special_tokens["bos"])
    eos_id = tokenizer.encode_single_token(tok_config.special_tokens["eos"])
    return tokenizer, bos_id, eos_id


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
async def generate_streaming(
    model: GPT2,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    bos_id: int,
    eos_id: int,
    context_length: int,
    msg: cl.Message,
) -> str:
    """Generate tokens one at a time, streaming to the Chainlit message."""
    # Encode prompt with BOS
    prompt_ids = [bos_id] + list(tokenizer.encode_ordinary(prompt))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_text = ""

    for _ in range(max_new_tokens):
        # Trim to context window
        if input_ids.shape[1] > context_length:
            input_ids = input_ids[:, -context_length:]

        logits = model(input_ids)
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)

            # Top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, 1)
                next_token = sorted_indices.gather(-1, sampled)
            else:
                next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode just the new token
        token_text = tokenizer.decode([next_token.item()])
        generated_text += token_text

        # Stream to the UI
        await msg.stream_token(token_text)

        # Stop on EOS
        if next_token.item() == eos_id:
            break

    return generated_text


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@cl.on_chat_start
async def on_chat_start():
    """Initialize the app: discover models, load default, show settings."""

    # Enable TF32
    torch.set_float32_matmul_precision("high")

    # Discover available runs
    runs = discover_runs()
    if not runs:
        await cl.Message(
            content="❌ **No trained models found!**\n\n"
            "Please train a model first:\n"
            "```bash\n"
            "uv run python scripts/training/train_gpt2.py \\\n"
            "    --model-config configs/models/gpt2-8k-2l-speed.yaml \\\n"
            "    --training-config configs/training/gpt2-8k-speed.yaml \\\n"
            "    --data-config configs/data/tinystories-8k-speed.yaml\n"
            "```"
        ).send()
        return

    run_names = list(runs.keys())
    default_run = run_names[0]

    # Load tokenizer
    tokenizer, bos_id, eos_id = load_tokenizer()
    cl.user_session.set("tokenizer", tokenizer)
    cl.user_session.set("bos_id", bos_id)
    cl.user_session.set("eos_id", eos_id)
    cl.user_session.set("runs", runs)

    # Load default model
    run_info = runs[default_run]
    model, config = load_model(run_info["checkpoint"], run_info["model_config"], DEVICE)
    cl.user_session.set("model", model)
    cl.user_session.set("model_config", config)
    cl.user_session.set("current_run", default_run)

    # Setup settings panel
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="model_run",
                label="🧠 Model Run",
                values=run_names,
                initial_value=default_run,
            ),
            cl.input_widget.Slider(
                id="temperature",
                label="🌡️ Temperature",
                initial=0.8,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            cl.input_widget.Slider(
                id="top_p",
                label="🎯 Top-P",
                initial=0.95,
                min=0.1,
                max=1.0,
                step=0.05,
            ),
            cl.input_widget.Slider(
                id="max_tokens",
                label="📏 Max New Tokens",
                initial=200,
                min=10,
                max=500,
                step=10,
            ),
        ]
    ).send()

    # Store default settings
    cl.user_session.set("temperature", 0.8)
    cl.user_session.set("top_p", 0.95)
    cl.user_session.set("max_tokens", 200)

    # Welcome message
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    await cl.Message(
        content=f"## 🧒 TinyStories Generator\n\n"
        f"**Model loaded**: `{default_run}`\n"
        f"- **Parameters**: {n_params:.1f}M\n"
        f"- **Context length**: {config.context_length}\n"
        f"- **Architecture**: GPT-2 ({config.n_layers}L, {config.d_model}d, {config.n_heads}H)\n"
        f"- **Device**: `{DEVICE}`\n\n"
        f"💡 **Type a story prompt** to generate text (e.g., \"Once upon a time\")\n\n"
        f"⚙️ Use the **Settings** panel (left sidebar) to adjust temperature, "
        f"top-p, max tokens, or switch between models.\n\n"
        f"---\n"
        f"*Available runs: {len(runs)}*"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings changes (model switch, temperature, etc.)."""
    cl.user_session.set("temperature", settings["temperature"])
    cl.user_session.set("top_p", settings["top_p"])
    cl.user_session.set("max_tokens", int(settings["max_tokens"]))

    # Check if model changed
    new_run = settings["model_run"]
    current_run = cl.user_session.get("current_run")

    if new_run != current_run:
        runs = cl.user_session.get("runs")
        run_info = runs[new_run]

        await cl.Message(content=f"🔄 Loading model `{new_run}`...").send()

        model, config = load_model(
            run_info["checkpoint"], run_info["model_config"], DEVICE
        )
        cl.user_session.set("model", model)
        cl.user_session.set("model_config", config)
        cl.user_session.set("current_run", new_run)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        await cl.Message(
            content=f"✅ **Model switched to** `{new_run}`\n"
            f"- **Parameters**: {n_params:.1f}M\n"
            f"- **Context length**: {config.context_length}\n"
            f"- **Architecture**: GPT-2 ({config.n_layers}L, {config.d_model}d, {config.n_heads}H)"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages: generate text from the prompt."""
    model = cl.user_session.get("model")
    config = cl.user_session.get("model_config")
    tokenizer = cl.user_session.get("tokenizer")
    bos_id = cl.user_session.get("bos_id")
    eos_id = cl.user_session.get("eos_id")
    temperature = cl.user_session.get("temperature")
    top_p = cl.user_session.get("top_p")
    max_tokens = cl.user_session.get("max_tokens")

    if model is None:
        await cl.Message(content="❌ No model loaded. Please check the runs/ directory.").send()
        return

    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()

    # Generate with streaming
    await generate_streaming(
        model=model,
        tokenizer=tokenizer,
        prompt=message.content,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=DEVICE,
        bos_id=bos_id,
        eos_id=eos_id,
        context_length=config.context_length,
        msg=msg,
    )

    # Finalize the message
    await msg.update()
