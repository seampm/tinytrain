# tinytrain — neural network training in C++, from scratch

A from-scratch neural network training framework in C++20. No PyTorch, no
autograd library — just a dynamic reverse-mode AD engine, neural network
modules, optimizers, and a Llama-architecture transformer, all written by hand.
Companion to [tinyinfer](https://github.com/seampm/tinyinfer): tinytrain
*teaches* models, tinyinfer *runs* them. A trained checkpoint exports to
safetensors with Hugging Face tensor names and loads directly into tinyinfer.

## What's inside

**Autograd engine** (`src/ops.cpp`, `include/tinytrain/tensor.h`)
- Dynamic computation graph with reverse-mode automatic differentiation
- 25 differentiable ops: elementwise/broadcast arithmetic, reductions, shape
  ops, matmul/batched matmul, embedding, softmax/log-softmax, cross-entropy,
  RMSNorm, dropout, RoPE
- Every op gradient verified by numerical finite-difference gradient checking

**Neural network modules** (`src/nn.cpp`)
- `Linear`, `Embedding`, `RMSNorm` with parameter management

**Optimizers** (`src/optim.cpp`)
- AdamW (decoupled weight decay) and SGD with momentum

**Transformer** (`src/model.cpp`)
- Llama-architecture GPT: RMSNorm, causal multi-head self-attention with RoPE,
  SwiGLU feed-forward, tied input/output embeddings
- Exports to safetensors + `config.json` using Hugging Face tensor names, so
  checkpoints load in tinyinfer (and any HF-compatible loader) with no conversion

**Tooling** (`apps/`)
- `prepare_data`: BPE-tokenize a text corpus (from `tokenizer.json`) to a binary
  token stream
- `train`: full training loop — AdamW, linear warmup + cosine decay, gradient
  clipping, checkpointing, loss logging

## Validation

- **22/22 unit tests pass**, including finite-difference gradient checks through
  the full transformer
- **Forward parity with tinyinfer**: the same checkpoint run through tinytrain
  and through tinyinfer (an independent implementation) agrees to 1.8e-6 —
  this check caught two real bugs (attention head layout, RMSNorm
  normalization) that gradient checking alone cannot see
- **Overfit test**: on a tiny repeating corpus, loss falls from 10.8 to 6.1 in
  30 steps, confirming end-to-end learning

## Training benchmark

13.3M-parameter Llama model (dim 256, 6 layers, 8 heads, vocab 32000),
batch 16 × seq 128, on a 2-core CPU:

| | tok/s |
|---|---|
| tinytrain (fp32) | ~125 |

The matmul uses OpenMP with cache-blocked tiling. No GPU, no BLAS — just
hand-written C++.

## Usage

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# tokenize
./build/prepare_data tokenizer.json corpus.txt train.tokens
# train
./build/train --tokens train.tokens --out run1 --steps 1500 \
    --batch-size 16 --seq-len 128 --lr 3e-4 --warmup 100 --seed 42
# run it: load run1/checkpoint in tinyinfer
```

## Training run

A 13M-parameter model is currently training on 3.1M TinyStories tokens
(1500 steps). Loss curve and samples to follow.

## Layout

```
include/tinytrain/   tensor.h, ops.h, nn.h, optim.h, model.h, tokenizer.h
src/                 tensor.cpp, ops.cpp, nn.cpp, optim.cpp, model.cpp, tokenizer.cpp
apps/                train.cpp, prepare_data.cpp
tests/               GoogleTest suite (autograd, nn/optim, model, scaffold)
```
