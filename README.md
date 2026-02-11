# TinyChatGPT (Starter)

TinyChatGPT is a from-scratch, decoder-only Transformer (GPT-style) chatbot project built to learn the full training pipeline end-to-end.
It starts small on CPU (laptop-friendly) and scales to GPU/cloud later using the same architecture and codebase.

## Goals
- Implement a real GPT-style model (causal self-attention) and train it on Q/A formatted text
- Keep the project modular: tokenizer ↔ dataset ↔ model ↔ training ↔ inference
- Scale safely from tiny CPU runs → Colab/Kaggle GPU runs → larger datasets/models

## Project Structure
- `src/model/` : decoder-only transformer + generation
- `src/data/`  : tokenizer, dataset, chat formatting
- `src/train/` : training & evaluation loops
- `src/inference/` : CLI chat + (future) FastAPI server
- `configs/` : CPU/GPU training configs
- `data/` : raw and processed datasets

## Quickstart (CPU)
1) Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
