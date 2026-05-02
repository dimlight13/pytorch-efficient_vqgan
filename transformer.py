# =============================================
# transformer.py  (Stage-2 per Efficient-VQGAN)
# =============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from efficient_vqgan import EfficientVQGAN

class BertishConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class FullSelfAttention(nn.Module):
    """ Multi-head self-attention WITHOUT causal mask (bidirectional) """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = FullSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BidirectionalTransformer(nn.Module):
    """Transformer backbone with bidirectional attention, used for MAE-style training"""
    def __init__(self, vocab_size, block_size, n_layer=24, n_head=16, n_embd=1024,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1):
        super().__init__()
        config = BertishConfig(vocab_size=vocab_size, block_size=block_size,
                               n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                               embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None):
        token_embeddings = self.tok_emb(idx)               # [B, Ttok, C]
        x = torch.cat((embeddings, token_embeddings), dim=1) if embeddings is not None else token_embeddings
        t = x.shape[1]
        assert t <= self.block_size, f"Sequence length {t} exceeds block_size {self.block_size}."
        x = self.drop(x + self.pos_emb[:, :t, :])
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# =========================================
# Stage-2: Multi-grained Transformer module
# =========================================

class VQGANTransformer(nn.Module):
    """
    Efficient-VQGAN Stage-2 (closer to paper):
    - Masked autoencoding training (bidirectional) on local windows
    - Global tokens from block-wise Conv2d over token embeddings
    - Block-wise autoregressive sampling; in-block parallel updates
    - Final token sequence FT = [GT, LT]  (no SOS)
    """
    def __init__(self, args):
        super().__init__()
        self.vocab_base = args.num_codebook_vectors
        self.mask_token = self.vocab_base           # [MASK]
        self.vocab_size = self.vocab_base + 1       # only MASK is extra

        self.vqgan = self.load_vqgan(args)
        self.vqgan.eval()
        for p in self.vqgan.parameters():
            p.requires_grad = False

        self.ws = getattr(args, "ws", 8)
        self.es = getattr(args, "es", 2)
        self.mask_ratio_default = getattr(args, "mask_ratio", 0.5)
        self.global_block_mask_ratio = getattr(args, "global_block_mask_ratio", 0.15)

        self.transformer = BidirectionalTransformer(
            vocab_size=self.vocab_size,
            block_size=getattr(args, "block_size", 1024),
            n_layer=getattr(args, "n_layer", 24),
            n_head=getattr(args, "n_head", 16),
            n_embd=getattr(args, "n_embd", 1024),
            embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
        )

        C = self.transformer.config.n_embd
        self.global_conv = nn.Conv2d(C, C, kernel_size=self.ws, stride=self.ws, padding=0, bias=True)

    @staticmethod
    def load_vqgan(args):
        model = EfficientVQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], quant_z.shape[2], quant_z.shape[3])
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices):
        if indices.dim() == 3:
            B, H, W = indices.shape
            flat = indices.reshape(B, H * W) 
            p1, p2 = H, W
        else:
            B, L = indices.shape
            p1 = p2 = int(math.sqrt(L)); assert p1 * p2 == L
            flat = indices
        emb = self.vqgan.codebook.embedding(flat).reshape(B, p1, p2, self.vqgan.codebook.D)
        emb = emb.permute(0, 3, 1, 2)
        return self.vqgan.decode(emb)

    def _token_emb_grid(self, token_grid):
        B, H, W = token_grid.shape
        tok = self.transformer.tok_emb(token_grid.reshape(B, -1))
        tok = tok.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return tok

    def _global_tokens_embeddings(self, token_grid):
        B, H, W = token_grid.shape
        emb_grid = self._token_emb_grid(token_grid)
        assert H % self.ws == 0 and W % self.ws == 0, f"Latent {(H, W)} must be divisible by ws={self.ws}."
        pooled = self.global_conv(emb_grid)  # [B, C, H/ws, W/ws]
        pooled = pooled.permute(0, 2, 3, 1).contiguous().view(B, -1, self.transformer.config.n_embd)
        return pooled

    def _random_block_mask(self, token_grid, block_mask_ratio=0.15):
        B, H, W = token_grid.shape
        ws = self.ws
        Hg, Wg = H // ws, W // ws
        mask_blocks = (torch.rand(B, Hg, Wg, device=token_grid.device) < block_mask_ratio)
        mask_up = mask_blocks.repeat_interleave(ws, dim=1).repeat_interleave(ws, dim=2)
        out = token_grid.clone()
        out[mask_up] = self.mask_token
        return out

    def _crop_with_extend(self, grid, bi, bj):
        B, H, W = grid.shape
        ws, es = self.ws, self.es
        rs, re = bi * ws, (bi + 1) * ws
        cs, ce = bj * ws, (bj + 1) * ws
        rs_e, cs_e = max(0, rs - es), max(0, cs - es)
        re_e, ce_e = min(H, re + es), min(W, ce + es)
        local = grid[:, rs_e:re_e, cs_e:ce_e]
        core_mask = torch.zeros_like(local, dtype=torch.bool)
        core_r0, core_c0 = rs - rs_e, cs - cs_e
        core_mask[:, core_r0:core_r0 + ws, core_c0:core_c0 + ws] = True
        return local, core_mask, (rs_e, re_e, cs_e, ce_e), (core_r0, core_c0)

    def _gather_local_batch_padded(self, grid, bi, bj):
        B, H, W = grid.shape
        ws, es = self.ws, self.es
        hf = wf = ws + 2 * es

        local = torch.full((B, hf, wf), self.mask_token, dtype=grid.dtype, device=grid.device)
        core_mask = torch.zeros((B, hf, wf), dtype=torch.bool, device=grid.device)
        valid_mask = torch.zeros((B, hf, wf), dtype=torch.bool, device=grid.device)

        for b in range(B):
            rs = int(bi[b].item()) * ws
            cs = int(bj[b].item()) * ws
            r0, c0 = max(0, rs - es), max(0, cs - es)
            r1, c1 = min(H, rs + ws + es), min(W, cs + ws + es)
            h, w = r1 - r0, c1 - c0

            local[b, 0:h, 0:w] = grid[b, r0:r1, c0:c1]
            valid_mask[b, 0:h, 0:w] = True

            core_r0 = rs - r0
            core_c0 = cs - c0
            core_mask[b, core_r0:core_r0 + ws, core_c0:core_c0 + ws] = True

        return local, core_mask, valid_mask

    def _mask_local(self, local, valid_mask, mask_ratio):
        prob = torch.full_like(local, fill_value=mask_ratio, dtype=torch.float32, device=local.device)
        mask = (torch.bernoulli(prob).bool()) & valid_mask
        masked = local.clone()
        masked[mask] = self.mask_token
        return masked, mask

    def forward(self, x, mask_ratio=None):
        with torch.no_grad():
            _, idx_grid = self.encode_to_z(x)          # [B,H,W]

        B, H, W = idx_grid.shape
        ws = self.ws
        assert H % ws == 0 and W % ws == 0, "Latent size must be divisible by ws."

        Hg, Wg = H // ws, W // ws
        bi = torch.randint(low=0, high=Hg, size=(B,), device=idx_grid.device)
        bj = torch.randint(low=0, high=Wg, size=(B,), device=idx_grid.device)

        mr = self.mask_ratio_default if mask_ratio is None else mask_ratio

        token_grid_for_global = self._random_block_mask(idx_grid, self.global_block_mask_ratio)
        global_emb = self._global_tokens_embeddings(token_grid_for_global)   # [B, Ng, C]

        local, core_mask, valid_mask = self._gather_local_batch_padded(idx_grid, bi, bj)  # [B,hf,wf]
        local_masked, local_mask_bool = self._mask_local(local, valid_mask, mr)           # [B,hf,wf]

        target_mask = local_mask_bool & core_mask & valid_mask                             # [B,hf,wf]

        L = local_masked.shape[1] * local_masked.shape[2]
        tokens_in = local_masked.reshape(B, L)                                             # [B,L]
        logits = self.transformer(tokens_in, embeddings=global_emb)                        # [B, Ng+L, V]
        Ng = global_emb.shape[1]
        logits_lt = logits[:, Ng:, :]                                                      # [B,L,V]

        targets_all = local.reshape(B, L)                                                  # [B,L]
        ignore = torch.full_like(targets_all, fill_value=-100)
        targets_ce = torch.where(target_mask.reshape(B, L), targets_all, ignore)           # [B,L]
        loss = F.cross_entropy(logits_lt.reshape(-1, self.vocab_size),
                            targets_ce.reshape(-1), ignore_index=-100)

        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def sample_blockwise(self, batch_size=4, H=16, W=16, steps_per_block=8, temperature=1.0, top_k=None, device="cuda"):
        ws = self.ws
        grid = torch.full((batch_size, H, W), self.mask_token, dtype=torch.long, device=device)
        Hg, Wg = H // ws, W // ws
        for bi in range(Hg):
            for bj in range(Wg):
                for _step in range(steps_per_block):
                    local, core_mask, coords, _ = self._crop_with_extend(grid, bi, bj)
                    B, h, w = local.shape
                    global_emb = self._global_tokens_embeddings(grid)
                    logits = self.transformer(local.view(B, -1), embeddings=global_emb)
                    Ng = global_emb.shape[1]
                    logits_lt = logits[:, Ng:, :] / max(temperature, 1e-8)
                    if top_k is not None:
                        v, _ = torch.topk(logits_lt, top_k)
                        kth = v[:, :, [-1]]
                        logits_lt[logits_lt < kth] = -float("inf")
                    probs = F.softmax(logits_lt, dim=-1)
                    preds = torch.argmax(probs, dim=-1).reshape(B, h, w)
                    mask_pos = (local == self.mask_token) & core_mask
                    local[mask_pos] = preds[mask_pos]
                    rs_e, re_e, cs_e, ce_e = coords
                    grid[:, rs_e:re_e, cs_e:ce_e] = local
        return grid

    @torch.no_grad()
    def log_images(self, x):
        logs = {}
        with torch.no_grad():
            _, idx_grid = self.encode_to_z(x)
            rec = self.z_to_image(idx_grid)
            logs["input"], logs["rec"] = x, rec

            B, H, W = idx_grid.shape
            half = idx_grid.clone(); half[:, :, W//2:] = self.mask_token
            filled = half.clone()
            Hg, Wg = H // self.ws, W // self.ws
            for bi in range(Hg):
                for bj in range(Wg):
                    rs, re = bi*self.ws, (bi+1)*self.ws
                    cs, ce = bj*self.ws, (bj+1)*self.ws
                    if ce <= W//2:  # left half known
                        continue
                    for _ in range(4):
                        local, core_mask, coords, _ = self._crop_with_extend(filled, bi, bj)
                        B2, h, w = local.shape
                        global_emb = self._global_tokens_embeddings(filled)
                        logits = self.transformer(local.view(B2, -1), embeddings=global_emb)
                        Ng = global_emb.shape[1]
                        logits_lt = logits[:, Ng:, :]
                        preds = torch.argmax(F.softmax(logits_lt, dim=-1), dim=-1).reshape(B2, h, w)
                        mask_pos = (local == self.mask_token)
                        local[mask_pos] = preds[mask_pos]
                        rs_e, re_e, cs_e, ce_e = coords
                        filled[:, rs_e:re_e, cs_e:ce_e] = local
            half_img = self.z_to_image(filled.reshape(B, -1))

            sampled_idx = self.sample_blockwise(batch_size=B, H=H, W=W, steps_per_block=8, device=x.device)
            full_img = self.z_to_image(sampled_idx.reshape(B, -1))
            logs["half_sample"], logs["full_sample"] = half_img, full_img

        concat = torch.cat([x, logs["rec"], logs["half_sample"], logs["full_sample"]], dim=0)
        return logs, concat

