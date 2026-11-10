"""ONNX-safe wrappers. Trained parameters are reused without approximation.

Mask construction bypasses Transformers' tracing-dependent mask API. It preserves
local/full attention and per-layer RoPE. Batch size is deliberately one.
"""
import torch
from torch import nn
import torch.nn.functional as F

class TextStack(nn.Module):
    """The checkpoint's transformer layers, run with explicitly built masks and rotary tables.

    Transformers builds attention masks through a tracing-dependent API; this class bypasses it and rebuilds the same bidirectional masks (each layer's local window, or full attention) and the per-layer RoPE
    tables once per call, then runs the original layers unchanged. `legacy` reads the fast checkpoint's per-layer
    `attention_windows` and `rope_theta` lists; otherwise `layer_types` and `rope_parameters`. VERIFIED: the exported
    graphs match the checkpoints within the errors recorded in documentation/VALIDATION.md and ADAPTER_VALIDATION.md."""
    def __init__(self, model, legacy=False):
        super().__init__()
        self.model, self.legacy = model, legacy
        cfg = model.config
        self.windows = cfg.attention_windows if legacy else [cfg.sliding_window if t == 'sliding_attention' else None for t in cfg.layer_types]
        self.bases = cfg.rope_theta if legacy else [cfg.rope_parameters[t]['rope_theta'] for t in cfg.layer_types]
        self.dim = cfg.head_dim

    def forward(self, embeddings, attention_mask):
        """`embeddings` (1, n, d) and a 0/1 `attention_mask` (1, n) to the final-normed hidden states (1, n, d)."""
        length = embeddings.shape[1]
        pos = torch.arange(length, device=embeddings.device)
        allowed = attention_mask[:, None, None, :].bool()
        full = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        masks, ropes = {None: full}, {}
        for window in set(self.windows) - {None}:
            # HF's bidirectional sliding mask includes +/- sliding_window.
            local = (pos[:, None] - pos[None, :]).abs() <= window
            masks[window] = torch.where(allowed & local[None, None], 0.0, torch.finfo(torch.float32).min)
        for base in set(self.bases):
            inv = 1.0 / base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=embeddings.device) / self.dim)
            freq = pos.float()[:, None] * inv[None, :]
            phase = torch.cat((freq, freq), -1)[None]
            ropes[base] = phase.cos(), phase.sin()
        h = embeddings
        for layer, window, base in zip(self.model.layers, self.windows, self.bases):
            h = layer(h, attention_mask=masks[window], position_embeddings=ropes[base])
        return self.model.norm(h)

class FastDecision(nn.Module):
    """v0.1 text decision head: cosine between the pooled state and each option token's projection (plus the
    projection of the hidden state at position 1), over the learned temperature, plus the option classifier. Option positions and the pooling mask come from the renderer (src/format.ts) as
    inputs, so the graph is plain tensor math; it returns one logit per option and JavaScript applies the softmax."""
    def __init__(self, decision):
        super().__init__()
        self.decision = decision
        self.stack = TextStack(decision.encoder, legacy=True)

    def forward(self, input_ids, attention_mask, option_positions, pool_mask):
        d = self.decision
        h = self.stack(d.encoder.embed_tokens(input_ids), attention_mask).float()
        pooled = (h * pool_mask[:, :, None]).sum(1) / pool_mask.sum(1, keepdim=True)
        option = h[0].index_select(0, option_positions)
        v = F.normalize(d.w_p(d.norm(pooled)), dim=-1)
        u = F.normalize(d.w_o(d.norm(option)) + d.w_q(d.norm(h[:, 1])), dim=-1)
        return (u * v).sum(-1) / d.log_tau.exp() + d.cls_mlp(option).squeeze(-1)

class VisionEncoder(nn.Module):
    """Vision tower plus connector: one 512 px tile to its image-token features. A graph of its own, so each tile is
    encoded once per request and its features are reused for every question."""
    def __init__(self, encoder):
        super().__init__()
        self.vision = encoder.vision_model
        self.connector = encoder.connector

    def forward(self, pixel_values):
        return self.connector(self.vision(pixel_values=pixel_values).last_hidden_state)

class VisionDecision(nn.Module):
    """v0.1 vision decision head. Image features replace the token embeddings where `image_indices` is nonzero
    (index k selects feature row k; 0 keeps the text embedding), the text stack runs, and each option's tokens are
    mean-pooled through `option_mask` into the head."""
    def __init__(self, encoder, head):
        super().__init__()
        self.stack = TextStack(encoder.text_model)
        self.head = head

    def forward(self, input_ids, attention_mask, image_features, image_indices, option_mask):
        h = self.stack.model.embed_tokens(input_ids)
        # One extra zero row allows text-only decisions without a fake image pass.
        features = torch.cat((torch.zeros_like(image_features[:1]), image_features), 0)
        visual = features.index_select(0, image_indices.reshape(-1)).reshape_as(h)
        h = torch.where((image_indices > 0)[:, :, None], visual, h)
        h = self.stack(h, attention_mask)
        pooled = torch.matmul(option_mask, h[0]) / option_mask.sum(-1, keepdim=True)
        return self.head(pooled).squeeze(-1)
