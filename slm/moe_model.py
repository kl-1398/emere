import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOBES = [
    "Brainstem",
    "Cerebellum",
    "Frontal Lobe",
    "Hippocampus",
    "Occipital Lobe",
    "Parietal Lobe",
    "Temporal Lobe",
    "Private",
]


class MoEConfig:
    vocab_size = 8000
    block_size = 256
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.1
    n_experts = len(LOBES)
    top_k = 2
    router_loss_weight = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Expert(nn.Module):
    """One expert = one lobe. Same shape as the dense model's MLP so a
    trained dense checkpoint's MLP weights can be copied in directly
    (sparse upcycling) instead of starting every expert from random init."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.router = nn.Linear(config.n_embd, config.n_experts)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.top_k = config.top_k

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)

        gate_logits = self.router(h)  # (B, T, E)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)  # (B, T, k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        expert_outs = torch.stack([e(h) for e in self.experts], dim=-2)  # (B, T, E, C)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, expert_outs.size(-1))
        chosen = torch.gather(expert_outs, dim=2, index=idx_exp)  # (B, T, k, C)
        moe_out = (chosen * topk_probs.unsqueeze(-1)).sum(dim=2)  # (B, T, C)

        x = x + moe_out
        return x, gate_logits


class TinyMoEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, lobe_labels=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        all_gate_logits = []
        for block in self.blocks:
            x, gate_logits = block(x)
            all_gate_logits.append(gate_logits)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = lm_loss
            if lobe_labels is not None:
                # Every token in a sequence shares that sequence's lobe label
                # (each training example is drawn entirely from one lobe's text).
                # Supervising the router directly is what keeps expert == lobe,
                # rather than letting the router freelance its own grouping.
                router_losses = []
                for gl in all_gate_logits:
                    Bg, Tg, Eg = gl.shape
                    target = lobe_labels.unsqueeze(1).expand(Bg, Tg).reshape(-1)
                    router_losses.append(F.cross_entropy(gl.reshape(-1, Eg), target))
                router_loss = torch.stack(router_losses).mean()
                loss = lm_loss + self.config.router_loss_weight * router_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, force_lobe=None):
        """force_lobe: optional int index into LOBES. Overrides the router so
        every token in every block is routed 100% to that lobe's expert —
        lets you literally ask the model to 'speak from Temporal Lobe' etc."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self._forward_with_override(idx_cond, force_lobe)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def _forward_with_override(self, idx, force_lobe):
        if force_lobe is None:
            return self.forward(idx)

        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = x + block.attn(block.ln_1(x))
            h = block.ln_2(x)
            x = x + block.experts[force_lobe](h)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None
