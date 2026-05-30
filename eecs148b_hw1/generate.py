"""
Text generation from a trained Transformer LM checkpoint.

Usage:
    uv run eecs148b_hw1/generate.py --checkpoint checkpoints/best.pt \
        --prompt "Once upon a time" --max_tokens 256 \
        --temperature 0.8 --top_p 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eecs148b_hw1.model import TransformerLM, softmax
from eecs148b_hw1.tokenizer import Tokenizer

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out all but the smallest set of tokens whose cumulative prob ≥ p."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens once cumulative probability exceeds p
    # (keep at least one token)
    remove = (cumulative - sorted_probs) >= p
    sorted_probs[remove] = 0.0
    # Scatter back to original ordering
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs)
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample one token from logits with temperature scaling and top-p filtering."""
    if temperature == 0.0:
        return int(logits.argmax().item())

    scaled = logits / temperature
    probs = softmax(scaled, dim=-1)

    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
        total = probs.sum()
        if total > 0:
            probs = probs / total

    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> str:
    model.eval()
    eot_id = tokenizer._bytes_to_id[b"<|endoftext|>"]

    token_ids = tokenizer.encode(prompt)
    context_length = model.context_length

    for _ in range(max_tokens):
        # Trim to context window
        input_ids = torch.tensor(
            [token_ids[-context_length:]], dtype=torch.long, device=device
        )
        logits = model(input_ids)          # (1, T, vocab_size)
        next_logits = logits[0, -1, :]     # last position

        next_id = sample_next_token(next_logits, temperature=temperature, top_p=top_p)
        token_ids.append(next_id)

        if next_id == eot_id:
            break

    return tokenizer.decode(token_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained Transformer LM")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["args"]

    tokenizer = Tokenizer.from_files(
        str(DATA_DIR / "vocab.json"),
        str(DATA_DIR / "merges.txt"),
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=4 * cfg["d_model"],
        use_layernorm=not cfg.get("no_layernorm", False),
        use_pos_emb=not cfg.get("no_pos_emb", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    print(f"Loaded checkpoint from step {ckpt['step']} (val loss {ckpt['val_loss']:.4f})")
    print(f"Prompt: {args.prompt!r}\n")

    output = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(output)


if __name__ == "__main__":
    main()
