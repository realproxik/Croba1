"""The Croba decoder-only transformer."""

from dataclasses import dataclass
import os
import torch
from torch import nn
from torch.nn import functional as F

try:
    import croba_cuda
except ImportError:
    croba_cuda = None


@dataclass
class CrobaConfig:
    vocab_size: int = 256
    context_length: int = 256
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    dropout: float = 0.1


class SquareReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_kernel = os.getenv("CROBA_USE_CUDA_KERNEL") == "1"
        if use_kernel and croba_cuda is not None and x.is_cuda:
            return _SquareReLUCUDA.apply(x)
        return F.relu(x).square()


class _SquareReLUCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return croba_cuda.square_relu(x.contiguous())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * F.relu(x)


class Block(nn.Module):
    def __init__(self, cfg: CrobaConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            SquareReLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, attn_mask=causal, need_weights=False)
        x = x + y
        return x + self.mlp(self.ln2(x))


class Croba(nn.Module):
    def __init__(self, cfg: CrobaConfig):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.size(1) > self.cfg.context_length:
            raise ValueError("sequence is longer than context_length")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.8):
        for _ in range(max_new_tokens):
            window = tokens[:, -self.cfg.context_length :]
            logits, _ = self(window)
            probs = F.softmax(logits[:, -1] / max(temperature, 1e-4), dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probs, 1)), dim=1)
        return tokens
