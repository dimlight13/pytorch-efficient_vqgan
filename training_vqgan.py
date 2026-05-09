import os
import re
import shutil
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lpips_vgg import LPIPS

from efficient_vqgan import EfficientVQGAN
from utils import load_data, weights_init, flexible_load_pretrained
from pg_modules.discriminator import ProjectedDiscriminator as Discriminator


class CudaPrefetcher:
    """Transfers the next batch to GPU while the current batch is being processed."""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self._next = None

    def _preload(self, it):
        try:
            imgs, labels = next(it)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            self._next = (
                imgs.to(self.device, non_blocking=True, memory_format=torch.channels_last),
                labels.to(self.device, non_blocking=True) if isinstance(labels, torch.Tensor) else labels,
            )

    def __iter__(self):
        it = iter(self.loader)
        self._preload(it)
        while self._next is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            imgs, labels = self._next
            imgs.record_stream(torch.cuda.current_stream())
            self._preload(it)
            yield imgs, labels

    def __len__(self):
        return len(self.loader)


class TrainEfficientVQGAN:
    def __init__(self, args):
        if args.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.vqgan = EfficientVQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(
            diffaug=bool(args.disc_diffaug_policy),
            diffaug_policy=args.disc_diffaug_policy,
            interp224=False,
            backbone_kwargs={"num_discs": args.disc_num_discs},
        ).to(device=args.device)

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

        self._ensure_decoder_refine_scale(getattr(args, "decoder_refine_init", 0.05))

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
        if args.compile and args.device.startswith("cuda"):
            if shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"):
                self.vqgan = torch.compile(self.vqgan, mode="reduce-overhead")
                self.discriminator = torch.compile(self.discriminator, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead mode). First few batches will be slow.")
            else:
                print("torch.compile skipped: no C compiler found (install gcc or set CC env var)")

        self.train(args)

    def _restore_resume_state(self):
        rs = self._resume_state
        if rs is None:
            return
        vq_optimizer_compatible = True
        ckpt_model = rs.get("model")
        if isinstance(ckpt_model, dict):
            current_model = getattr(self.vqgan, "_orig_mod", self.vqgan).state_dict()
            mismatched = [
                k for k, v in ckpt_model.items()
                if k in current_model and current_model[k].shape != v.shape
            ]
            if mismatched:
                vq_optimizer_compatible = False
                print("[Resume] checkpoint model shapes differ from current architecture; "
                      "skipping VQ optimizer/scheduler restore.")
                print(f"[Resume] first mismatched tensor: {mismatched[0]}")
        restored_disc = False
        if rs.get("discriminator") is not None:
            try:
                self.discriminator.load_state_dict(rs["discriminator"], strict=False)
                restored_disc = True
            except Exception as e:
                print(f"[Resume] could not restore discriminator: {e}")
        elif bool(rs.get("beta1_switched", False)):
            print("[Resume] checkpoint has GAN state but no discriminator weights; "
                  "skipping discriminator optimizer/scheduler restore.")
        for key, target in (
            ("opt_vq", self.opt_vq),
            ("opt_disc", self.opt_disc),
            ("sched_vq", self.sched_vq),
            ("sched_disc", self.sched_disc),
        ):
            if key in {"opt_vq", "sched_vq"} and not vq_optimizer_compatible:
                continue
            if key in {"opt_disc", "sched_disc"} and not restored_disc and bool(rs.get("beta1_switched", False)):
                continue
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
            f"[Resume] restored state "
            f"(vq_optimizer={vq_optimizer_compatible}, discriminator={restored_disc}, "
            f"beta1_switched={self._beta1_switched}, global_step={self._resume_global_step})"
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

    @staticmethod
    def scheduled_perceptual_size(args, global_step):
        if (
            args.perceptual_warmup_image_size > 0
            and global_step < args.perceptual_fullres_start
        ):
            return args.perceptual_warmup_image_size
        return args.perceptual_image_size

    @staticmethod
    def scheduled_max_gan_weight(args, global_step):
        if args.max_effective_gan_weight <= 0:
            return args.max_effective_gan_weight
        return EfficientVQGAN.adopt_weight(
            args.max_effective_gan_weight,
            global_step,
            threshold=args.disc_start,
            value=args.max_effective_gan_weight_start,
            rampup_steps=args.gan_weight_rampup_steps,
        )

    def _ensure_decoder_refine_scale(self, init_scale):
        if init_scale <= 0:
            return
        nudged = 0
        for module in self.vqgan.decoder.refine.modules():
            if not hasattr(module, "res_scale"):
                continue
            with torch.no_grad():
                if module.res_scale.detach().abs().max().item() < init_scale * 0.25:
                    module.res_scale.fill_(init_scale)
                    nudged += 1
        if nudged:
            print(f"[Init] primed {nudged} decoder refine scale(s) to {init_scale}")

    def save_checkpoint(self, path, epoch, global_step):
        state = {
            "model": getattr(self.vqgan, "_orig_mod", self.vqgan).state_dict(),
            "discriminator": getattr(self.discriminator, "_orig_mod", self.discriminator).state_dict(),
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
        if args.device.startswith("cuda"):
            train_loader = CudaPrefetcher(train_loader, args.device)
        steps_per_epoch = len(train_loader)
        accum_steps = args.gradient_accumulation_steps
        defer_codebook_update = accum_steps > 1

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

        self.opt_vq.zero_grad(set_to_none=True)
        self.opt_disc.zero_grad(set_to_none=True)

        for epoch in range(self.start_epoch, args.epochs):
            for i, (imgs, _) in enumerate(tqdm(train_loader)):
                global_step = epoch * steps_per_epoch + i
                is_last_in_window = (
                    (i + 1) % accum_steps == 0 or (i + 1) == steps_per_epoch
                )

                disc_factor = self.vqgan.adopt_weight(
                    args.disc_factor,
                    global_step,
                    threshold=args.disc_start,
                    rampup_steps=args.disc_rampup_steps,
                )
                disc_loss_factor = self.vqgan.adopt_weight(
                    args.disc_loss_factor,
                    global_step,
                    threshold=args.disc_start,
                    rampup_steps=args.disc_rampup_steps,
                )
                disc_active = disc_factor > 0 or disc_loss_factor > 0
                if disc_active and not self._beta1_switched:
                    self._set_beta1(self.opt_vq, args.beta1)
                    self._beta1_switched = True
                    print(
                        f"\n[Step {global_step}] Discriminator activated "
                        f"(G factor={disc_factor:.4g}, D factor={disc_loss_factor:.4g}) — "
                        "expect slower iterations from here."
                    )

                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    decoded_images, _, q_loss = self.vqgan(
                        imgs,
                        global_step=global_step,
                        defer_codebook_update=defer_codebook_update,
                    )

                    # Reconstruction + Perceptual
                    use_perceptual = args.perceptual_loss_factor > 0 and (
                        args.perceptual_every <= 1 or global_step % args.perceptual_every == 0
                    )
                    if use_perceptual:
                        perceptual_image_size = self.scheduled_perceptual_size(args, global_step)
                        imgs_lpips, decoded_lpips = self.resize_for_perceptual(
                            imgs, decoded_images, perceptual_image_size
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
                            d_weight = self.vqgan.calculate_lambda(
                                nll_loss,
                                g_loss,
                                max_weight=args.max_adaptive_gan_weight,
                            )
                        except RuntimeError:
                            d_weight = torch.tensor(0.0, device=args.device)

                        gan_weight = d_weight * disc_factor
                        max_gan_weight = self.scheduled_max_gan_weight(args, global_step)
                        if max_gan_weight > 0:
                            gan_weight = torch.clamp(gan_weight, max=max_gan_weight)

                        vq_loss = vq_loss + gan_weight * g_loss

                        for p in self.discriminator_params:
                            p.requires_grad = True

                vq_loss = vq_loss / accum_steps

                # Backward G
                if self.scaler.is_enabled():
                    self.scaler.scale(vq_loss).backward()
                else:
                    vq_loss.backward()

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
                        gan_loss = disc_loss_factor * 0.5 * (d_loss_real + d_loss_fake) / accum_steps

                    if self.scaler.is_enabled():
                        self.scaler.scale(gan_loss).backward()
                    else:
                        gan_loss.backward()

                if is_last_in_window:
                    if defer_codebook_update:
                        self.vqgan.codebook.flush_pending_ema_update()

                    if self.scaler.is_enabled():
                        self.scaler.step(self.opt_vq)
                        if disc_active:
                            self.scaler.step(self.opt_disc)
                        self.scaler.update()
                    else:
                        self.opt_vq.step()
                        if disc_active:
                            self.opt_disc.step()
                    self.sched_vq.step()
                    if disc_active:
                        self.sched_disc.step()
                    self.opt_vq.zero_grad(set_to_none=True)
                    self.opt_disc.zero_grad(set_to_none=True)

                if (
                    args.dead_code_reset_every > 0
                    and global_step > args.freeze_codebook_steps
                    and global_step % args.dead_code_reset_every == 0
                ):
                    n_reset = self.vqgan.codebook.reset_dead_codes()
                    if n_reset:
                        print(f"\n[Step {global_step}] reset {n_reset} dead codebook entries")

                if i % 1000 == 0:
                    with torch.no_grad():
                        real_imgs_denorm = (imgs[:4] + 1) / 2
                        fake_imgs_denorm = (decoded_images[:4] + 1) / 2
                        real_fake_images = torch.cat((real_imgs_denorm, fake_imgs_denorm))
                        out_path = os.path.join("results", f"{epoch}_{i}.jpg")
                        vutils.save_image(real_fake_images, out_path, nrow=4)

            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                mem_gb = torch.cuda.memory_allocated() / 1024 ** 3
                reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
                print(f"[Epoch {epoch}] CUDA memory: {mem_gb:.1f} GB allocated / {reserved_gb:.1f} GB reserved")

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
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=4.5e-5)
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1 once GAN training is active (and for the discriminator)')
    parser.add_argument('--beta1-pre-disc', type=float, default=0.9,
                        help='Adam beta1 for the VQ optimizer before disc-start (faster pre-GAN convergence)')
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--warmup-steps', type=int, default=5000,
                        help='Linear LR warmup over this many steps for both optimizers')
    parser.add_argument('--disc-start', type=int, default=25000)
    parser.add_argument('--disc-rampup-steps', type=int, default=20000,
                        help='Linearly ramp discriminator factor over this many steps after disc-start; set 0 to disable')
    parser.add_argument('--disc-factor', type=float, default=0.5,
                        help='Generator-side adversarial factor after disc-start')
    parser.add_argument('--disc-loss-factor', type=float, default=1.0,
                        help='Discriminator-side hinge loss factor after disc-start')
    parser.add_argument('--max-adaptive-gan-weight', type=float, default=100.0,
                        help='Clamp adaptive VQGAN GAN weight d_weight; set <=0 to disable this clamp')
    parser.add_argument('--max-effective-gan-weight-start', type=float, default=1.0,
                        help='Initial effective GAN weight clamp at disc-start')
    parser.add_argument('--max-effective-gan-weight', type=float, default=2.0,
                        help='Final clamp for d_weight * disc_factor before applying generator GAN loss; set <=0 to disable')
    parser.add_argument('--gan-weight-rampup-steps', type=int, default=50000,
                        help='Ramp max effective GAN weight from start to final after disc-start')
    parser.add_argument('--disc-diffaug-policy', type=str, default='color,translation',
                        help='DiffAugment policy for discriminator; empty string disables DiffAugment')
    parser.add_argument('--disc-num-discs', type=int, default=2, choices=[1, 2, 3, 4],
                        help='Number of projected discriminator feature scales to use')
    parser.add_argument('--rec-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0)
    parser.add_argument('--perceptual-image-size', type=int, default=256,
                        help='Final LPIPS input size; set 0 to use full training resolution')
    parser.add_argument('--perceptual-warmup-image-size', type=int, default=128,
                        help='LPIPS input size before perceptual-fullres-start; set <=0 to disable warmup size')
    parser.add_argument('--perceptual-fullres-start', type=int, default=50000,
                        help='Global step at which LPIPS switches from warmup size to perceptual-image-size')
    parser.add_argument('--perceptual-every', type=int, default=1,
                        help='Compute LPIPS every N steps; 1 keeps the perceptual loss active every step')
    parser.add_argument('--encoder-stem', type=str, default='hybrid', choices=['hybrid', 'patch4'],
                        help='Encoder input stem: hybrid uses two stride-2 convs; patch4 uses one non-overlap stride-4 conv')
    parser.add_argument('--encoder-pre-quant-blocks', type=int, default=1,
                        help='Residual conv blocks before quant_conv')
    parser.add_argument('--decoder-min-upsample-channels', type=int, default=64,
                        help='Minimum channel count kept after decoder PatchExpanding stages')
    parser.add_argument('--decoder-refine-blocks', type=int, default=2,
                        help='Number of lightweight full-resolution decoder refinement blocks')
    parser.add_argument('--decoder-refine-init', type=float, default=0.05,
                        help='Initial residual scale for decoder refinement blocks')
    parser.add_argument('--freeze-codebook-steps', type=int, default=5000,
                        help='Disable codebook EMA updates before this global step')
    parser.add_argument('--codebook-update-interval', type=int, default=1,
                        help='Update codebook EMA every N steps after freeze-codebook-steps')
    parser.add_argument('--codebook-ema-decay', type=float, default=0.99,
                        help='EMA decay for codebook embeddings')
    parser.add_argument('--codebook-eps', type=float, default=1e-5,
                        help='Numerical epsilon for EMA codebook normalization')
    parser.add_argument('--codebook-lookup-chunk-size', type=int, default=8192,
                        help='Chunk size for nearest-code lookup; set <=0 for full distance matrix')
    parser.add_argument('--dead-code-threshold', type=float, default=0.1,
                        help='EMA cluster size below which a code is considered dead and reset')
    parser.add_argument('--dead-code-reset-every', type=int, default=2000,
                        help='Reset dead codes every N global steps; set 0 to disable')

    parser.add_argument('--checkpoint-path', type=str, default="checkpoints/vqgan_epoch_1.pt",
                        help='Path to checkpoint file to resume training')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader worker processes for parallel data loading')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Accumulate gradients over N mini-batches before each optimizer step. '
                             'Effective batch size = batch_size * N. Use to simulate multi-GPU effective batch size.')
    parser.add_argument('--compile', action='store_true', default=True,
                        help='Enable torch.compile (reduce-overhead mode) for faster training on CUDA')
    parser.add_argument('--no-fused-window-process', dest='fused_window_process', action='store_false',
                        help='Disable native fused Swin window processing when the CUDA extension is available')
    parser.set_defaults(fused_window_process=True)

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
