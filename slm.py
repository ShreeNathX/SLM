"""A small decoder-only language model used by the notebook and Streamlit app."""

import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, context: int, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(context, context, dtype=torch.bool), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        normalized = self.attn_norm(x)
        y, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=self.causal_mask[:t, :t],
            need_weights=False,
        )
        x = x + y
        return x + self.mlp(self.mlp_norm(x))


class TinyDecoder(nn.Module):
    """Randomly initialized GPT-style decoder with tied input/output embeddings."""

    def __init__(self, config: dict):
        super().__init__()
        width = config["n_embd"]
        self.context = config["block_size"]
        self.token_embedding = nn.Embedding(config["vocab_size"], width)
        self.position_embedding = nn.Embedding(self.context, width)
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(width, config["n_head"], self.context, config["dropout"])
                for _ in range(config["n_layer"])
            ]
        )
        self.final_norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, length = tokens.shape
        if length > self.context:
            raise ValueError(f"Sequence length {length} exceeds context {self.context}.")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.final_norm(self.blocks(x))
        # The output projection shares the randomly initialized token embeddings.
        logits = F.linear(x, self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, end_token_id: int, max_new_tokens=80,
             temperature=0.9, top_k=40) -> str:
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    prompt_limit = max(1, model.context - max_new_tokens)
    encoded = encoded[-prompt_limit:] or [end_token_id]
    tokens = torch.tensor([encoded], dtype=torch.long, device=next(model.parameters()).device)
    prompt_length = tokens.size(1)

    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -model.context:])
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k:
            cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        if next_token.item() == end_token_id:
            break
        tokens = torch.cat((tokens, next_token), dim=1)

    return tokenizer.decode(tokens[0, prompt_length:].tolist())
