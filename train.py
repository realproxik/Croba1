"""CUDA-only training entry point for Croba."""

import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from .model import Croba, CrobaConfig
from .tokenizer import ByteTokenizer


def batch(data: torch.Tensor, size: int, context: int, device: str):
    starts = torch.randint(0, len(data) - context - 1, (size,))
    x = torch.stack([data[i : i + context] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + context + 1] for i in starts]).to(device)
    return x, y


def main() -> None:
    p = argparse.ArgumentParser(description="Train Croba on a CUDA GPU")
    p.add_argument("--data", required=True, help="UTF-8 training text")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--out", default="checkpoints/croba.pt")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "Croba training requires CUDA. Open Croba_Colab.ipynb in Colab and select a GPU runtime."
        )
    device = "cuda"
    raw = Path(args.data).read_text(encoding="utf-8")
    data = torch.tensor(ByteTokenizer.encode(raw), dtype=torch.long)
    if len(data) <= args.context + 1:
        raise SystemExit(f"Dataset needs more than {args.context + 1} encoded bytes.")

    cfg = CrobaConfig(context_length=args.context)
    model = Croba(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {torch.cuda.get_device_name(0)}")
    for step in range(1, args.steps + 1):
        x, y = batch(data, args.batch_size, args.context, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "model": model.state_dict()}, out)
    print(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    main()

