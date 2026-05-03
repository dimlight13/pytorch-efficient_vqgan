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
        if args.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.vqgan = EfficientVQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(diffaug=True, interp224=False).to(device=args.device)

        for p in self.discriminator.feature_network.parameters():
            p.requires_grad = False
        self.discriminator_params = list(self.discriminator.discriminator.parameters())
        self.discriminator.discriminator.apply(weights_init)
        self.discriminator.train()
        self.perceptual_loss = LPIPS(use_dropout=False).eval().to(device=args.device)

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

        if args.device.startswith("cuda"):
            self.vqgan.to(memory_format=torch.channels_last)
            self.discriminator.to(memory_format=torch.channels_last)
            self.perceptual_loss.to(memory_format=torch.channels_last)

        self.use_amp = args.device.startswith("cuda")
        self.amp_device_type = "cuda" if self.use_amp else "cpu"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        adam_kwargs = {
            "lr": lr,
            "eps": 1e-08,
            "betas": (args.beta1, args.beta2),
        }
        if args.device.startswith("cuda"):
            adam_kwargs["fused"] = True

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

            opt_vq = torch.optim.Adam(params, **adam_kwargs)
        else:
            opt_vq = torch.optim.Adam(
                [
                    {"params": group_loaded, "lr": lr * 0.2},
                    {"params": group_new,    "lr": lr},
                ],
                **adam_kwargs
            )

        opt_disc = torch.optim.Adam(
            self.discriminator_params, **adam_kwargs)
        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    @staticmethod
    def resize_for_perceptual(imgs, decoded_images, size):
        if size <= 0 or imgs.shape[-2:] == (size, size):
            return imgs, decoded_images
        imgs_lpips = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        decoded_lpips = F.interpolate(decoded_images, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        return imgs_lpips, decoded_lpips

    def train(self, args):
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        train_loader = load_data(args)
        steps_per_epoch = len(train_loader)

        global_step = 0
        for epoch in range(args.epochs):
            for i, (imgs, _) in enumerate(tqdm(train_loader)):

                imgs = imgs.to(
                    device=args.device,
                    non_blocking=args.device.startswith("cuda"),
                    memory_format=torch.channels_last,
                )
                global_step = epoch * steps_per_epoch + i

                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    # Reconstruction + Perceptual
                    use_perceptual = args.perceptual_loss_factor > 0 and (
                        args.perceptual_every <= 1 or global_step % args.perceptual_every == 0
                    )
                    if use_perceptual:
                        imgs_lpips, decoded_lpips = self.resize_for_perceptual(
                            imgs, decoded_images, args.perceptual_image_size
                        )
                        perceptual_loss = self.perceptual_loss(imgs_lpips, decoded_lpips)
                    else:
                        perceptual_loss = decoded_images.new_zeros(())
                    rec_loss = (imgs - decoded_images) ** 2

                    nll_loss = (args.rec_loss_factor * rec_loss +
                               args.perceptual_loss_factor * perceptual_loss)
                    nll_loss = nll_loss.mean()

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, global_step, threshold=args.disc_start)
                    disc_active = disc_factor > 0

                    vq_loss = (
                        nll_loss
                        + q_loss.mean()
                    )

                    if disc_active:
                        for p in self.discriminator_params:
                            p.requires_grad = False

                        disc_fake_for_g = self.discriminator(decoded_images, None)
                        g_loss = -torch.mean(disc_fake_for_g)

                        try:
                            d_weight = self.vqgan.calculate_lambda(nll_loss, g_loss)
                        except RuntimeError:
                            d_weight = torch.tensor(0.0, device=args.device)

                        vq_loss = vq_loss + d_weight * disc_factor * g_loss

                        for p in self.discriminator_params:
                            p.requires_grad = True

                # Backward
                self.opt_vq.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(vq_loss).backward()
                    self.scaler.step(self.opt_vq)
                else:
                    vq_loss.backward()
                    self.opt_vq.step()

                if disc_active:
                    # =============================
                    # Discriminator (hinge loss)
                    # =============================
                    with torch.amp.autocast(
                        device_type=self.amp_device_type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        disc_fake_for_d = self.discriminator(decoded_images.detach(), None)
                        disc_real = self.discriminator(imgs.detach(), None)
                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake_for_d))
                        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_disc.zero_grad(set_to_none=True)
                    if self.scaler.is_enabled():
                        self.scaler.scale(gan_loss).backward()
                        self.scaler.step(self.opt_disc)
                    else:
                        gan_loss.backward()
                        self.opt_disc.step()

                if self.scaler.is_enabled():
                    self.scaler.update()

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
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-codebook-vectors', type=int, default=1024)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image-channels', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name (imagenet or cifar10)')
    parser.add_argument('--dataset-path', type=str, default='./data', help='ImageNet dataset root path')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=4e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--disc-start', type=int, default=50000)
    parser.add_argument('--disc-factor', type=float, default=1.0)
    parser.add_argument('--rec-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-image-size', type=int, default=128,
                        help='LPIPS input size; set 0 to use full training resolution')
    parser.add_argument('--perceptual-every', type=int, default=1,
                        help='Compute LPIPS every N steps; 1 keeps the perceptual loss active every step')
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
