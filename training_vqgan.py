import os
import re
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
        self._resume_state = None
        self.start_epoch = 0
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint (flex) from {args.checkpoint_path}")
            raw_ckpt = torch.load(args.checkpoint_path, map_location=args.device)
            if isinstance(raw_ckpt, dict) and "epoch" in raw_ckpt:
                self._resume_state = raw_ckpt
                self.start_epoch = int(raw_ckpt["epoch"]) + 1
                print(f"[Resume] rich checkpoint detected: last completed epoch={raw_ckpt['epoch']}, "
                      f"resuming at epoch={self.start_epoch}")
            else:
                m = re.search(r"epoch[_-]?(\d+)", os.path.basename(args.checkpoint_path))
                if m:
                    self.start_epoch = int(m.group(1)) + 1
                    print(f"[Resume] inferred from filename: resuming at epoch={self.start_epoch}")
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
        self.sched_vq = self._make_warmup_scheduler(self.opt_vq, args.warmup_steps)
        self.sched_disc = self._make_warmup_scheduler(self.opt_disc, args.warmup_steps)
        self._beta1_switched = False
        self._resume_global_step = None
        self._restore_resume_state()

        self.prepare_training()
        self.train(args)

    def _restore_resume_state(self):
        rs = self._resume_state
        if rs is None:
            return
        for key, target in (
            ("opt_vq", self.opt_vq),
            ("opt_disc", self.opt_disc),
            ("sched_vq", self.sched_vq),
            ("sched_disc", self.sched_disc),
        ):
            if key in rs and rs[key] is not None:
                try:
                    target.load_state_dict(rs[key])
                except Exception as e:
                    print(f"[Resume] could not restore {key}: {e}")
        if rs.get("scaler") is not None and self.scaler.is_enabled():
            try:
                self.scaler.load_state_dict(rs["scaler"])
            except Exception as e:
                print(f"[Resume] could not restore scaler: {e}")
        self._beta1_switched = bool(rs.get("beta1_switched", False))
        if "global_step" in rs:
            self._resume_global_step = int(rs["global_step"])
        print(
            f"[Resume] restored optimizer/scheduler state "
            f"(beta1_switched={self._beta1_switched}, global_step={self._resume_global_step})"
        )

    def configure_optimizers(self, args):
        lr = args.learning_rate
        # Pre-GAN phase uses beta1=0.9 for fast convergence; switched to args.beta1
        # (typically 0.5) once the discriminator kicks in for GAN stability.
        vq_adam_kwargs = {
            "lr": lr,
            "eps": 1e-08,
            "betas": (args.beta1_pre_disc, args.beta2),
        }
        disc_adam_kwargs = {
            "lr": lr,
            "eps": 1e-08,
            "betas": (args.beta1, args.beta2),
        }
        if args.device.startswith("cuda"):
            vq_adam_kwargs["fused"] = True
            disc_adam_kwargs["fused"] = True

        group_loaded, group_new = [], []
        for n, p in self.vqgan.named_parameters():
            if not p.requires_grad:
                continue
            if n in self.loaded_param_names:
                group_loaded.append(p)
            else:
                group_new.append(p)

        if len(self.loaded_param_names) == 0:
            trainable_modules = (
                list(self.vqgan.encoder.parameters())
                + list(self.vqgan.decoder.parameters())
                + list(self.vqgan.quant_conv.parameters())
                + list(self.vqgan.post_quant_conv.parameters())
            )
            params = [p for p in trainable_modules if p.requires_grad]

            opt_vq = torch.optim.Adam(params, **vq_adam_kwargs)
        else:
            opt_vq = torch.optim.Adam(
                [
                    {"params": group_loaded, "lr": lr * 0.2},
                    {"params": group_new,    "lr": lr},
                ],
                **vq_adam_kwargs
            )

        opt_disc = torch.optim.Adam(
            self.discriminator_params, **disc_adam_kwargs)
        return opt_vq, opt_disc

    @staticmethod
    def _set_beta1(opt, new_beta1):
        for group in opt.param_groups:
            _, b2 = group["betas"]
            group["betas"] = (new_beta1, b2)

    @staticmethod
    def _make_warmup_scheduler(opt, warmup_steps):
        warmup_steps = max(1, int(warmup_steps))

        def lr_lambda(step):
            return min(1.0, (step + 1) / warmup_steps)

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

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

    def save_checkpoint(self, path, epoch, global_step):
        state = {
            "model": self.vqgan.state_dict(),
            "opt_vq": self.opt_vq.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
            "sched_vq": self.sched_vq.state_dict(),
            "sched_disc": self.sched_disc.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "beta1_switched": bool(self._beta1_switched),
        }
        torch.save(state, path)

    def train(self, args):
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        train_loader = load_data(args)
        steps_per_epoch = len(train_loader)

        if self.start_epoch >= args.epochs:
            print(f"[Resume] start_epoch={self.start_epoch} >= epochs={args.epochs}; nothing to do.")
            return
        if self.start_epoch > 0:
            print(f"[Resume] training will continue from epoch {self.start_epoch}")

        global_step = (
            self._resume_global_step
            if self._resume_global_step is not None
            else self.start_epoch * steps_per_epoch
        )
        for epoch in range(self.start_epoch, args.epochs):
            for i, (imgs, _) in enumerate(tqdm(train_loader)):

                imgs = imgs.to(
                    device=args.device,
                    non_blocking=args.device.startswith("cuda"),
                    memory_format=torch.channels_last,
                )
                global_step = epoch * steps_per_epoch + i

                disc_factor = self.vqgan.adopt_weight(
                    args.disc_factor,
                    global_step,
                    threshold=args.disc_start,
                    rampup_steps=args.disc_rampup_steps,
                )
                disc_active = disc_factor > 0
                if disc_active and not self._beta1_switched:
                    self._set_beta1(self.opt_vq, args.beta1)
                    self._beta1_switched = True

                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    decoded_images, _, q_loss = self.vqgan(imgs, global_step=global_step)

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
                    rec_loss = (imgs - decoded_images).abs()

                    nll_loss = (args.rec_loss_factor * rec_loss +
                               args.perceptual_loss_factor * perceptual_loss)
                    nll_loss = nll_loss.mean()

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
                self.sched_vq.step()

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
                    self.sched_disc.step()

                if self.scaler.is_enabled():
                    self.scaler.update()

                if (
                    args.dead_code_reset_every > 0
                    and global_step > args.freeze_codebook_steps
                    and global_step % args.dead_code_reset_every == 0
                ):
                    self.vqgan.codebook.reset_dead_codes()

                if i % 1000 == 0:
                    with torch.no_grad():
                        real_imgs_denorm = (imgs[:4] + 1) / 2
                        fake_imgs_denorm = (decoded_images[:4] + 1) / 2
                        real_fake_images = torch.cat((real_imgs_denorm, fake_imgs_denorm))
                        out_path = os.path.join("results", f"{epoch}_{i}.jpg")
                        vutils.save_image(real_fake_images, out_path, nrow=4)

            ckpt_path = os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt")
            self.save_checkpoint(ckpt_path, epoch=epoch, global_step=global_step)


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
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=4.5e-5)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 once GAN training is active (and for the discriminator)')
    parser.add_argument('--beta1-pre-disc', type=float, default=0.9,
                        help='Adam beta1 for the VQ optimizer before disc-start (faster pre-GAN convergence)')
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--warmup-steps', type=int, default=5000,
                        help='Linear LR warmup over this many steps for both optimizers')
    parser.add_argument('--disc-start', type=int, default=50000)
    parser.add_argument('--disc-rampup-steps', type=int, default=10000,
                        help='Linearly ramp discriminator factor over this many steps after disc-start; set 0 to disable')
    parser.add_argument('--disc-factor', type=float, default=1.0)
    parser.add_argument('--rec-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-image-size', type=int, default=128,
                        help='LPIPS input size; set 0 to use full training resolution')
    parser.add_argument('--perceptual-every', type=int, default=1,
                        help='Compute LPIPS every N steps; 1 keeps the perceptual loss active every step')
    parser.add_argument('--decoder-refine-blocks', type=int, default=1,
                        help='Number of lightweight full-resolution decoder refinement blocks')
    parser.add_argument('--freeze-codebook-steps', type=int, default=200,
                        help='Disable codebook EMA updates before this global step')
    parser.add_argument('--codebook-update-interval', type=int, default=1,
                        help='Update codebook EMA every N steps after freeze-codebook-steps')
    parser.add_argument('--codebook-ema-decay', type=float, default=0.99,
                        help='EMA decay for codebook embeddings')
    parser.add_argument('--codebook-eps', type=float, default=1e-5,
                        help='Numerical epsilon for EMA codebook normalization')
    parser.add_argument('--dead-code-threshold', type=float, default=0.5,
                        help='EMA cluster size below which a code is considered dead and reset')
    parser.add_argument('--dead-code-reset-every', type=int, default=500,
                        help='Reset dead codes every N global steps; set 0 to disable')

    parser.add_argument('--checkpoint-path', type=str, default=None,
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
