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
        C = 48                              # base C after patch partition
        self.patch_embed = nn.Conv2d(args.image_channels, C, 4, 4)

        H = args.image_size // 4
        W = args.image_size // 4

        depths = [2, 2, 2]
        heads  = [3, 6, 12]
        ws     = [8, 4, 2]

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

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        C = 48
        dim_start = 4 * C                 # 4C
        self.in_proj = nn.Linear(args.latent_dim, dim_start)

        H = args.image_size // 16
        W = args.image_size // 16

        self.residual_block = ResidualBlock(
            dim_in=dim_start, dim_out=dim_start * 2, hw=(H, W),
            resi_connection='3conv'
        )

        dim = dim_start * 2               # 8C

        depths = [2, 2, 2]
        heads  = [12, 6, 3]
        window_sizes = [2, 4, 8]

        self.upsample_blocks = nn.ModuleList()
        for i in range(3):
            ws_eff = max(1, min(window_sizes[i], H, W))

            swin_blocks = nn.ModuleList()
            for j in range(depths[i]):
                shift = (ws_eff // 2) if (j % 2 == 1 and ws_eff > 1) else 0
                swin_blocks.append(
                    SwinBlock(dim, (H, W), heads[i], ws_eff, shift)
                )

            patch_expanding = PatchExpanding(dim)   # Expand: 채널 1/2, 해상도 ×2

            self.upsample_blocks.append(nn.ModuleDict({
                "swin_blocks": swin_blocks,
                "patch_expanding": patch_expanding
            }))

            dim = dim // 2
            H, W = H * 2, W * 2

        self.final_feat_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),   # H/2 -> H
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.to_rgb = nn.Conv2d(C, args.image_channels, 1)

    def forward(self, z):
        B, C_latent, H, W = z.shape  # H=W=latent 해상도 (H_img/16)

        x = z.flatten(2).transpose(1, 2)   # (B, HW, D)
        x = self.in_proj(x)                # (B, HW, 4C)
        x = self.residual_block(x, H, W)         # (B, HW, 8C)

        curH, curW = H, W                  # H/16, W/16

        for block in self.upsample_blocks:
            for swin in block["swin_blocks"]:
                x = swin(x)
            x, curH, curW = block["patch_expanding"](x, curH, curW)

        x = x.view(B, curH, curW, -1).permute(0, 3, 1, 2).contiguous()

        x = self.final_feat_up(x)          # (B, C, H, W)
        x = self.to_rgb(x)                 # (B, 3, H, W)
        return x

class CodebookEMA(nn.Module):
    def __init__(self, args, decay=0.99, eps=1e-5, update=True):
        super().__init__()
        self.K = args.num_codebook_vectors
        self.D = args.latent_dim
        self.beta = args.beta
        self.decay, self.eps = decay, eps
        self.update = update

        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embedding.weight, -1 / self.K, 1 / self.K)
        self.embedding.weight.requires_grad = False

        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", torch.zeros(self.K, self.D))
        self._ema_inited = False

    @torch.no_grad()
    def _maybe_init_ema_from_embedding(self):
        if not self._ema_inited:
            self.cluster_size.fill_(1.0)
            self.embed_avg.copy_(self.embedding.weight.data)
            self._ema_inited = True

    @torch.no_grad()
    def _assign_indices(self, flat: torch.Tensor):
        w = self.embedding.weight
        x2 = (flat ** 2).sum(1, keepdim=True)        # (N,1)
        w2 = (w ** 2).sum(1)                          # (K,)
        dist = x2 + w2.unsqueeze(0) - 2 * flat @ w.t()# (N,K)
        return torch.argmin(dist, dim=1)              # (N,)

    def forward(self, z: torch.Tensor):
        # (B,C,H,W) → (N,D)
        B, C, H, W = z.shape
        z_nhwc = z.permute(0, 2, 3, 1).contiguous()
        flat = z_nhwc.view(-1, self.D)

        self._maybe_init_ema_from_embedding()

        with torch.no_grad():
            idx = self._assign_indices(flat)  # (N,)

        z_q = self.embedding(idx).view_as(z_nhwc)

        if self.training and self.update:
            with torch.no_grad():
                counts = torch.bincount(idx, minlength=self.K).to(self.cluster_size.dtype)  # (K,)
                embed_sum = torch.zeros_like(self.embed_avg)                                # (K,D)
                embed_sum.index_add_(0, idx, flat)

                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n     # (K,)
                embed_mean = self.embed_avg / smoothed.unsqueeze(1)                         # (K,D)
                self.embedding.weight.data.copy_(embed_mean)

        # commitment loss (EMA VQ 관례)
        loss = self.beta * F.mse_loss(z_nhwc, z_q.detach())

        # straight-through
        z_q = z_nhwc + (z_q - z_nhwc).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, idx, loss


class EfficientVQGAN(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)
        self.codebook=CodebookEMA(args)

        self.quant_conv = nn.Conv2d(self.encoder.out_dim, args.latent_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1)
        self.last_layer = self.decoder.to_rgb

    def forward(self,imgs):
        z=self.encoder(imgs); zq=self.quant_conv(z)
        zq,idx,qloss=self.codebook(zq)
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
    def adopt_weight(disc_factor, step, threshold, value=0.):
        """discriminator warm-up"""
        return disc_factor if step >= threshold else value

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.to_rgb
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))