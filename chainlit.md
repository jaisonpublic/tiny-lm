# 🧒 TinyStories Generator

Welcome to the **TinyStories Generator**! This app lets you interact with tiny GPT-2 language models trained from scratch on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

## How to Use

1. **Type a prompt** in the chat box (e.g., "Once upon a time there was a little")
2. The model will **stream** a story continuation token-by-token
3. Use the **⚙️ Settings** panel to adjust:
   - 🧠 **Model Run** — switch between different trained models
   - 🌡️ **Temperature** — higher = more creative, lower = more focused
   - 🎯 **Top-P** — nucleus sampling threshold
   - 📏 **Max Tokens** — maximum number of tokens to generate

## Example Prompts

- `Once upon a time`
- `There was a little girl named Lily who`
- `The big red dog was very`
- `One day, a bear went to the`

## About

These models are trained with the [tiny-lm](https://github.com/ferjorosa/tiny-lm) project — a learning-focused repository for pre-training small language models from scratch.
