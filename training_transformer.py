
# =============================================
# training_transformer.py  
# =============================================

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
import math

class TrainTransformer:
    def __init__(self, args):
        self.args = args
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        WL = (torch.nn.Linear, )
        BL = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, WL):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BL):
                    no_decay.add(fpn)

        no_decay.update({"pos_emb", "tok_emb.weight"})
        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        decay_keys    = decay    & set(param_dict.keys())
        no_decay_keys = no_decay & set(param_dict.keys())

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay_keys)],    "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(no_decay_keys)], "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=(0.9, 0.95))

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    @torch.no_grad()
    def save_stage12_preview(self, imgs, step):
        self.model.eval()
        try:
            x = imgs[:4].to(self.args.device)
            _, idx_grid = self.model.encode_to_z(x)           # [B, H, W]
            x_rec = self.model.z_to_image(idx_grid)            # [B, 3, H*, W*]

            grid = torch.cat([x, x_rec], dim=0)
            if grid.min() < 0:
                grid = (grid + 1) / 2

            os.makedirs("results/transformer", exist_ok=True)
            out_path = os.path.join("results/transformer", f"stage12_preview_step_{step:07d}.jpg")
            vutils.save_image(grid, out_path, nrow=4)
        finally:
            self.model.train()

    def train(self, args):
        self.prepare_training()
        train_loader = load_data(args)
        steps_per_epoch = len(train_loader)
        total_steps = args.epochs * steps_per_epoch
        global_step = 0

        max_mask = 0.9
        min_mask = getattr(args, "min_mask_ratio", 0.0) 
        for epoch in range(args.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for imgs, _ in pbar:
                imgs = imgs.to(device=args.device)
                # Cosine mask schedule gamma(r) = cos(pi/2 * r)
                r = min(1.0, global_step / max(1, total_steps - 1))
                mask_ratio = float(min_mask + (max_mask - min_mask) * math.cos(0.5 * math.pi * r))

                self.optim.zero_grad()
                loss, _ = self.model(imgs, mask_ratio=mask_ratio)
                loss.backward()
                self.optim.step()

                if global_step % 1000 == 0:
                    self.save_stage12_preview(imgs, global_step)

                pbar.set_postfix(MAE_Loss=float(loss.detach().cpu().item()), mask_ratio=round(mask_ratio, 3))
                global_step += 1

            with torch.no_grad():
                log, sampled_imgs = self.model.log_images(imgs[:4])
                sampled_imgs_denorm = (sampled_imgs + 1) / 2 if sampled_imgs.min() < 0 else sampled_imgs
                vutils.save_image(sampled_imgs_denorm, os.path.join("results", f"transformer_epoch_{epoch}.jpg"), nrow=4)
                plot_images(log)
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage-2 Transformer (Efficient-VQGAN style)")
    parser.add_argument('--latent-dim', type=int, default=1024)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-codebook-vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image-channels', type=int, default=3)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2.25e-04)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--ws', type=int, default=8)
    parser.add_argument('--es', type=int, default=2)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--global-block-mask-ratio', type=float, default=0.15)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--n-layer', type=int, default=12)
    parser.add_argument('--n-head', type=int, default=12)
    parser.add_argument('--n-embd', type=int, default=768)
    args = parser.parse_args()
    TrainTransformer(args)
