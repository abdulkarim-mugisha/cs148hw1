"""
Training loop for the Transformer LM on TinyStories.

Usage:
    uv run eecs148b_hw1/train.py [options]

Example:
    uv run eecs148b_hw1/train.py --vocab_size 10000 --d_model 512 \
        --num_layers 4 --num_heads 8 --context_length 256 \
        --batch_size 64 --total_steps 2500 --lr 3e-4 \
        --checkpoint_dir checkpoints/ --log_interval 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eecs148b_hw1.model import TransformerLM, cross_entropy
from eecs148b_hw1.tokenizer import Tokenizer

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.stack([
        torch.from_numpy(dataset[s : s + context_length].astype(np.int64))
        for s in starts
    ]).to(device)
    y = torch.stack([
        torch.from_numpy(dataset[s + 1 : s + context_length + 1].astype(np.int64))
        for s in starts
    ]).to(device)
    return x, y


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = args.device
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_files(
        str(DATA_DIR / "vocab.json"),
        str(DATA_DIR / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # ── Load datasets (memory-mapped) ────────────────────────────────────
    train_data = np.memmap(str(DATA_DIR / "train.bin"), dtype=np.uint16, mode="r")
    valid_data = np.memmap(str(DATA_DIR / "valid.bin"), dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,}  |  Valid tokens: {len(valid_data):,}")

    # ── Build model ───────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=4 * args.d_model,
        use_layernorm=not args.no_layernorm,
        use_pos_emb=not args.no_pos_emb,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # ── Resume from checkpoint if requested ──────────────────────────────
    start_step = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from step {ckpt['step']} (val loss {ckpt['val_loss']:.4f})")

    # ── Log file ─────────────────────────────────────────────────────────
    log_path = checkpoint_dir / "log.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    def log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    if not args.resume:
        log({"event": "start", "args": vars(args), "n_params": n_params})

    # ── Training loop ─────────────────────────────────────────────────────
    t0 = time.time()
    # Number of steps in this training phase (handles fresh start and resume)
    phase_steps = args.total_steps - start_step + 1

    for step in range(start_step, args.total_steps + 1):
        # Use step relative to this phase so resumed runs get a fresh LR schedule
        relative_step = step - start_step + 1
        lr = get_lr(relative_step, args.warmup_steps, phase_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        # logits: (B, T, V) — flatten batch and time dims for cross entropy
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tokens_seen = step * args.batch_size * args.context_length

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(args.val_batches):
                    vx, vy = get_batch(valid_data, args.batch_size, args.context_length, device)
                    vlogits = model(vx)
                    val_losses.append(cross_entropy(vlogits.view(-1, vocab_size), vy.view(-1)).item())
                val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(val_loss)

            print(
                f"step {step:5d}/{args.total_steps} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f} | ppl {val_ppl:.1f} | "
                f"lr {lr:.2e} | tokens {tokens_seen:,} | {elapsed:.0f}s"
            )
            log({
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "lr": lr,
                "tokens": tokens_seen,
                "elapsed_s": elapsed,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = checkpoint_dir / "best.pt"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  → checkpoint saved ({ckpt_path})")

    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM on TinyStories")
    # Model
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Logging / checkpointing
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Ablation flags
    p.add_argument("--no_layernorm", action="store_true", help="Remove all LayerNorms (ablation 1)")
    p.add_argument("--no_pos_emb", action="store_true", help="Remove positional embeddings / NoPE (ablation 2)")
    # Resume
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
