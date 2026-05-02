import os
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lpips import LPIPS

from efficient_vqgan import EfficientVQGAN
from utils import load_data, weights_init, flexible_load_pretrained
from pg_modules.discriminator import ProjectedDiscriminator as Discriminator

class TrainEfficientVQGAN:
    def __init__(self, args):
        self.vqgan = EfficientVQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(diffaug=True, interp224=False).to(device=args.device)

        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)

        self.loaded_param_names = set()
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint (flex) from {args.checkpoint_path}")
            self.loaded_param_names = flexible_load_pretrained(
                self.vqgan,
                ckpt_path=args.checkpoint_path,
                do_depth_surgery=True,
                interp_rpb=True,
                device=args.device,
                verbose=True
            )
            print("Checkpoint loaded with flexible mapping.")
        else:
            print("No checkpoint found or checkpoint path not specified. Starting from scratch.")

        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate

        group_loaded, group_new = [], []
        for n, p in self.vqgan.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.loaded_param_names:
                group_loaded.append(p)
            else:
                group_new.append(p)

        if len(self.loaded_param_names) == 0:
            params = (
                list(self.vqgan.encoder.parameters()) +
                list(self.vqgan.decoder.parameters()) +
                list(self.vqgan.codebook.parameters()) +
                list(self.vqgan.quant_conv.parameters()) +
                list(self.vqgan.post_quant_conv.parameters())
            )

            opt_vq = torch.optim.Adam(params, lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        else:
            opt_vq = torch.optim.Adam(
                [
                    {"params": group_loaded, "lr": lr * 0.2},
                    {"params": group_new,    "lr": lr},
                ],
                lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
            )

        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        train_loader = load_data(args)
        steps_per_epoch = len(train_loader)

        global_step = 0
        for epoch in range(args.epochs):
            for i, (imgs, _) in enumerate(tqdm(train_loader)):

                imgs = imgs.to(device=args.device)
                global_step = epoch * steps_per_epoch + i

                frozen = (global_step < args.freeze_codebook_steps)
                interval_ok = (global_step % args.codebook_update_interval == 0)
                self.vqgan.codebook.update = (not frozen) and interval_ok

                decoded_images, _, q_loss = self.vqgan(imgs)

                # Discriminator forward
                disc_real = self.discriminator(imgs, None)
                disc_fake_for_g = self.discriminator(decoded_images, None)
                disc_factor = self.vqgan.adopt_weight(args.disc_factor, global_step, threshold=args.disc_start)

                # Reconstruction + Perceptual
                perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                rec_loss = torch.abs(imgs - decoded_images)

                nll_loss = (args.rec_loss_factor * rec_loss + 
                           args.perceptual_loss_factor * perceptual_loss)
                nll_loss = nll_loss.mean()

                disc_fake_for_g = self.discriminator(decoded_images, None)
                g_loss = -torch.mean(disc_fake_for_g)

                try:
                    d_weight = self.vqgan.calculate_lambda(nll_loss, g_loss)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=args.device)

                disc_factor = self.vqgan.adopt_weight(args.disc_factor, global_step, threshold=args.disc_start)

                vq_loss = (
                    nll_loss
                    + d_weight * disc_factor * g_loss
                    + q_loss.mean()
                )

                # =============================
                # Discriminator (hinge loss)
                # =============================
                disc_fake_for_d = self.discriminator(decoded_images.detach(), None)
                disc_real = self.discriminator(imgs.detach(), None)
                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake_for_d))
                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                # Backward
                self.opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                self.opt_disc.zero_grad()
                gan_loss.backward()

                self.opt_vq.step()
                self.opt_disc.step()

                if i % 1000 == 0:
                    with torch.no_grad():
                        real_imgs_denorm = (imgs[:4] + 1) / 2
                        fake_imgs_denorm = (decoded_images[:4] + 1) / 2
                        real_fake_images = torch.cat((real_imgs_denorm, fake_imgs_denorm))
                        out_path = os.path.join("results", f"{epoch}_{i}.jpg")
                        vutils.save_image(real_fake_images, out_path, nrow=4)

            ckpt_path = os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt")
            torch.save(self.vqgan.state_dict(), ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN on ImageNet")
    parser.add_argument('--latent-dim', type=int, default=1024)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-codebook-vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image-channels', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (imagenet or cifar10)')
    parser.add_argument('--dataset-path', type=str, default='./data', help='ImageNet dataset root path')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=4.5e-6)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--disc-start', type=int, default=25000)
    parser.add_argument('--disc-factor', type=float, default=1.0)
    parser.add_argument('--rec-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.5)
    parser.add_argument('--freeze-codebook-steps', type=int, default=1000,
                        help='deactivate codebook EMA until the freeze step')
    parser.add_argument('--codebook-update-interval', type=int, default=1,
                        help='Update codebook EMA with per N step (1 is update every step)')


    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/conv3_vqgan_pretrain.pt',
                        help='Path to checkpoint file to resume training')

    args = parser.parse_args()

    # Check CUDA availability and set device accordingly
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    elif args.device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU device")
    
    train_vqgan = TrainEfficientVQGAN(args)
