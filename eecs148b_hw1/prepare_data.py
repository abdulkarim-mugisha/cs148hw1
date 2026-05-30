"""
Train a BPE tokenizer on TinyStories, then encode both splits to uint16 numpy arrays.

Usage:
    uv run eecs148b_hw1/prepare_data.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from eecs148b_hw1.tokenizer import Tokenizer, train_bpe

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_TXT = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_TXT = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

VOCAB_OUT = DATA_DIR / "vocab.json"
MERGES_OUT = DATA_DIR / "merges.txt"
TRAIN_BIN = DATA_DIR / "train.bin"
VALID_BIN = DATA_DIR / "valid.bin"


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    # Vocab: {str(id): list[int]} so JSON can serialise bytes cleanly
    vocab_json = {str(k): list(v) for k, v in vocab.items()}
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)

    with open(MERGES_OUT, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(a.decode("latin-1") + " " + b.decode("latin-1") + "\n")

    print(f"  Vocab saved to   {VOCAB_OUT}")
    print(f"  Merges saved to  {MERGES_OUT}")


def encode_split(tokenizer: Tokenizer, txt_path: Path, bin_path: Path) -> None:
    print(f"  Encoding {txt_path.name} …", flush=True)
    ids: list[int] = []
    with open(txt_path, encoding="utf-8") as f:
        for chunk in tokenizer.encode_iterable(f):
            ids.append(chunk)

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"    {len(arr):,} tokens → {bin_path.name}  ({arr.nbytes / 1e6:.1f} MB)")


def main() -> None:
    # ── Step 1: Train BPE tokenizer ───────────────────────────────────────────
    print(f"\n[1/3] Training BPE tokenizer (vocab_size={VOCAB_SIZE}) …")
    t0 = time.time()
    vocab, merges = train_bpe(TRAIN_TXT, VOCAB_SIZE, SPECIAL_TOKENS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  |  vocab size = {len(vocab)}")

    longest = max(vocab.values(), key=len)
    print(f"  Longest token ({len(longest)} bytes): {longest!r}")

    # ── Step 2: Serialise vocab and merges to disk ────────────────────────────
    print("\n[2/3] Saving vocab and merges …")
    save_vocab_and_merges(vocab, merges)

    # ── Step 3: Encode both splits ────────────────────────────────────────────
    print("\n[3/3] Encoding dataset splits …")
    tokenizer = Tokenizer(vocab, merges, SPECIAL_TOKENS)
    encode_split(tokenizer, TRAIN_TXT, TRAIN_BIN)
    encode_split(tokenizer, VALID_TXT, VALID_BIN)

    print("\nDone! All artefacts written to data/")


if __name__ == "__main__":
    main()
