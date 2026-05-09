# efficient_vqgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_module.swin_transformer import SwinTransformerBlock as SwinBlock, PatchMerging, PatchExpanding, PatchEmbed, PatchUnEmbed
from typing import Optional
from typing import Optional, Tuple

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        C = 64                              # base C after patch partition
        self.patch_embed = nn.Conv2d(args.image_channels, C, 4, 4)
        self.patch_norm = nn.LayerNorm(C)

        H = args.image_size // 4
        W = args.image_size // 4

        depths = [2, 6, 12]
        heads  = [2, 4, 8]                  # head_dim = 32 throughout
        ws     = [8, 8, 8]

        self.stages = nn.ModuleList()
        dim = C
        for i in range(3):
            ws_eff = max(1, min(ws[i], H, W))
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                shift = (ws_eff // 2) if (j % 2 == 1 and ws_eff > 1) else 0
                blocks.append(SwinBlock(dim, (H, W), heads[i], ws_eff, shift))

            merge = PatchMerging((H, W), dim) if i < 2 else None
            self.stages.append(nn.ModuleDict({"blocks": blocks, "merge": merge}))

            if merge is not None:
                dim *= 2
                H //= 2; W //= 2

        self.out_dim = dim
        self.final_H = H
        self.final_W = W

    def forward(self, x):
        x = self.patch_embed(x)                       # (B,C,H/4,W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)              # (B,HW,C)
        x = self.patch_norm(x)
        curH, curW = H, W

        for stage in self.stages:
            for swin in stage["blocks"]:
                x = swin(x)
            if stage["merge"] is not None:
                x = stage["merge"](x)
                curH //= 2
                curW //= 2

        x = x.transpose(1, 2).contiguous().view(B, self.out_dim, curH, curW)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hw: Tuple[int, int],
                 resi_connection: str = '1conv'):
        super().__init__()
        H, W = hw
        assert H > 0 and W > 0

        self.proj_in = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.patch_unembed = PatchUnEmbed(img_size=(H, W), patch_size=1, in_chans=dim_out, embed_dim=dim_out, norm_layer=None)
        self.patch_embed   = PatchEmbed(img_size=(H, W), patch_size=1, in_chans=dim_out, embed_dim=dim_out, norm_layer=None)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim_out, dim_out // 4, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_out // 4, dim_out // 4, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_out // 4, dim_out, 3, 1, 1),
            )
        else:
            raise ValueError("resi_connection must be '1conv' or '3conv'")

        self.H = H
        self.W = W

    def forward(self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        B, L, _ = x.shape
        H = H if H is not None else self.H
        W = W if W is not None else self.W
        if H * W != L:
            side = int(L ** 0.5)
            H, W = side, side

        x_proj = self.proj_in(x)                         # (B, HW, dim_out)
        feat = self.patch_unembed(x_proj, (H, W))        # (B, dim_out, H, W)
        feat = self.conv(feat)                           # conv residual in image domain
        feat = self.patch_embed(feat)                    # (B, HW, dim_out)
        return x_proj + feat


class DecoderRefineBlock(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = max(16, channels // bottleneck_ratio)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, hidden, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.block(x)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        C = 64
        dim_start = 4 * C                 # 4C = 256
        self.in_proj = nn.Linear(args.latent_dim, dim_start)

        H = args.image_size // 16
        W = args.image_size // 16

        self.residual_block = ResidualBlock(
            dim_in=dim_start, dim_out=dim_start, hw=(H, W),
            resi_connection='3conv'
        )

        dim = dim_start                   # 4C

        depths = [12, 6, 2]
        heads  = [8, 4, 2]                # head_dim = 32 throughout
        window_sizes = [8, 8, 8]

        self.upsample_blocks = nn.ModuleList()
        for i in range(3):
            ws_eff = max(1, min(window_sizes[i], H, W))

            swin_blocks = nn.ModuleList()
            for j in range(depths[i]):
                shift = (ws_eff // 2) if (j % 2 == 1 and ws_eff > 1) else 0
                swin_blocks.append(
                    SwinBlock(dim, (H, W), heads[i], ws_eff, shift)
                )

            patch_expanding = PatchExpanding(dim)   # 채널 1/2, 해상도 ×2

            self.upsample_blocks.append(nn.ModuleDict({
                "swin_blocks": swin_blocks,
                "patch_expanding": patch_expanding
            }))

            dim = dim // 2
            H, W = H * 2, W * 2

        final_dim = dim                              # = C/2 (32 with C=64)
        wide = 2 * C                                 # 128
        self.final_feat_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(final_dim, wide, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(wide, wide, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        refine_blocks = max(0, int(getattr(args, "decoder_refine_blocks", 1)))
        self.refine = nn.Sequential(*[DecoderRefineBlock(wide) for _ in range(refine_blocks)])

        self.to_rgb = nn.Conv2d(wide, args.image_channels, 3, padding=1)

    def forward(self, z):
        B, C_latent, H, W = z.shape  # H=W=latent 해상도 (H_img/16)

        x = z.flatten(2).transpose(1, 2)   # (B, HW, D)
        x = self.in_proj(x)                # (B, HW, 4C)
        x = self.residual_block(x, H, W)         # (B, HW, 4C)

        curH, curW = H, W                  # H/16, W/16

        for block in self.upsample_blocks:
            for swin in block["swin_blocks"]:
                x = swin(x)
            x, curH, curW = block["patch_expanding"](x, curH, curW)

        x = x.view(B, curH, curW, -1).permute(0, 3, 1, 2).contiguous()

        x = self.final_feat_up(x)          # (B, C, H, W)
        x = self.refine(x)
        x = self.to_rgb(x)                 # (B, 3, H, W)
        return x

class Codebook(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.K = args.num_codebook_vectors
        self.D = args.latent_dim
        self.beta = args.beta
        self.freeze_codebook_steps = getattr(args, "freeze_codebook_steps", 0)
        self.codebook_update_interval = max(1, getattr(args, "codebook_update_interval", 1))
        self.ema_decay = getattr(args, "codebook_ema_decay", 0.99)
        self.eps = getattr(args, "codebook_eps", 1e-5)
        self.dead_code_threshold = getattr(args, "dead_code_threshold", 1.0)

        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.D ** -0.5)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("ema_cluster_size", torch.ones(self.K))
        self.register_buffer("ema_w", self.embedding.weight.detach().clone())

    def _codebook_update_enabled(self, global_step: Optional[int]):
        if global_step is None:
            return True
        if global_step < self.freeze_codebook_steps:
            return False
        return (global_step - self.freeze_codebook_steps) % self.codebook_update_interval == 0

    @torch.no_grad()
    def _assign_indices(self, flat: torch.Tensor):
        flat = flat.float()
        w = self.embedding.weight.float()
        x2 = (flat ** 2).sum(1, keepdim=True)        # (N,1)
        w2 = (w ** 2).sum(1)                          # (K,)
        dist = x2 + w2.unsqueeze(0) - 2 * flat @ w.t()# (N,K)
        return torch.argmin(dist, dim=1)              # (N,)

    @torch.no_grad()
    def sync_ema_buffers(self):
        self.ema_cluster_size.fill_(1.0)
        self.ema_w.copy_(self.embedding.weight.detach())

    @torch.no_grad()
    def reset_dead_codes(self, perturb_std: float = 0.01):
        """Re-seed under-used codes with perturbed copies of active ones."""
        dead = self.ema_cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        active = ~dead
        n_active = int(active.sum().item())
        if n_active == 0:
            return 0
        active_w = self.embedding.weight[active]
        idx = torch.randint(0, n_active, (n_dead,), device=active_w.device)
        chosen = active_w[idx]
        noise = torch.randn_like(chosen) * perturb_std
        new_codes = chosen + noise
        self.embedding.weight[dead] = new_codes
        self.ema_w[dead] = new_codes.to(self.ema_w.dtype)
        avg_active = self.ema_cluster_size[active].mean().clamp(min=self.dead_code_threshold)
        self.ema_cluster_size[dead] = avg_active
        return n_dead

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor):
        flat = flat.float()
        cluster_size = flat.new_zeros(self.K)
        cluster_size.index_add_(0, idx, flat.new_ones(idx.shape[0]))
        embed_sum = flat.new_zeros(self.K, self.D)
        embed_sum.index_add_(0, idx, flat)

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_w.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.eps)
            / (n + self.K * self.eps)
            * n
        )
        embedding = self.ema_w / cluster_size.unsqueeze(1)
        self.embedding.weight.copy_(embedding.to(self.embedding.weight.dtype))

    def forward(self, z: torch.Tensor, global_step: Optional[int] = None):
        # (B,C,H,W) → (N,D)
        B, C, H, W = z.shape
        z_nhwc = z.permute(0, 2, 3, 1).contiguous()
        flat = z_nhwc.view(-1, self.D)

        with torch.no_grad():
            idx = self._assign_indices(flat)  # (N,)

        z_q = self.embedding(idx).view_as(z_nhwc)

        if self.training and self._codebook_update_enabled(global_step):
            self._ema_update(flat.detach(), idx)

        commitment_loss = self.beta * F.mse_loss(z_nhwc.float(), z_q.detach().float())
        loss = commitment_loss

        # straight-through
        z_q = z_nhwc + (z_q - z_nhwc).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, idx, loss


class EfficientVQGAN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)
        self.codebook=Codebook(args)

        self.quant_conv = nn.Conv2d(self.encoder.out_dim, args.latent_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)
        self.last_layer = self.decoder.to_rgb

    def forward(self, imgs, global_step: Optional[int] = None):
        z=self.encoder(imgs); zq=self.quant_conv(z)
        zq,idx,qloss=self.codebook(zq, global_step=global_step)
        zq=self.post_quant_conv(zq)
        rec=self.decoder(zq)
        return rec,idx,qloss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    @staticmethod
    def adopt_weight(disc_factor, step, threshold, value=0., rampup_steps=0):
        """discriminator warm-up"""
        if step < threshold:
            return value
        if rampup_steps <= 0:
            return disc_factor
        progress = min(1.0, float(step - threshold + 1) / float(rampup_steps))
        return value + (disc_factor - value) * progress

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.to_rgb
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return λ

    def load_checkpoint(self, path):
        incompatible = self.load_state_dict(torch.load(path), strict=False)
        missing = set(getattr(incompatible, "missing_keys", []))
        if {"codebook.ema_cluster_size", "codebook.ema_w"} & missing:
            self.codebook.sync_ema_buffers()
