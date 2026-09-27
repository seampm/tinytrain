#!/bin/bash
# Generate samples from a trained tinytrain checkpoint via tinyinfer
# Usage: ./gen_samples.sh <checkpoint_dir> <output_file>
set -e
CKPT_DIR="$1"
OUT="$2"
TMPDIR=$(mktemp -d)
cp "$CKPT_DIR/model.safetensors" "$TMPDIR/"
cp "$CKPT_DIR/config.json" "$TMPDIR/"
cp ~/workspace/tinyinfer/models/tinyllama-15m/tokenizer.json "$TMPDIR/"

cd ~/workspace/tinyinfer
{
  echo "=== Prompt: 'Once upon a time' (seed 42) ==="
  ./build/tinyinfer prompt "Once upon a time" --model "$TMPDIR" --max-tokens 80 --seed 42 2>&1 | head -5
  echo ""
  echo "=== Prompt: 'The little girl' (seed 123) ==="
  ./build/tinyinfer prompt "The little girl" --model "$TMPDIR" --max-tokens 80 --seed 123 2>&1 | head -5
  echo ""
  echo "=== Prompt: 'One day' (seed 7) ==="
  ./build/tinyinfer prompt "One day" --model "$TMPDIR" --max-tokens 80 --seed 7 2>&1 | head -5
} > "$OUT" 2>&1
rm -rf "$TMPDIR"
echo "Samples written to $OUT"
