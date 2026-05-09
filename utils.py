import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import math
import torch
import torch.nn.functional as F
import re
# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size
        
        # Filter for common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_files = os.listdir(path)
        self.images = [
            os.path.join(path, file) 
            for file in all_files 
            if os.path.splitext(file.lower())[1] in image_extensions
        ]
        self._length = len(self.images)
        
        if self._length == 0:
            raise ValueError(f"No image files found in {path}. Found files: {all_files}")

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            image = self.preprocessor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
            image = image.transpose(2, 0, 1)
            return image
        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {str(e)}")

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    """
    Adaptive data loader that switches between datasets based on args.dataset
    Supported datasets: 'imagenet', 'cifar10'
    """
    # Get dataset name from args (default to 'cifar10' if not specified)
    dataset_name = getattr(args, 'dataset', 'cifar10').lower()
    
    if dataset_name == 'imagenet':
        # ImageNet configuration
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        # ImageNet train data path
        train_path = os.path.join(args.dataset_path, 'imagenet', 'ILSVRC', 'Data', 'CLS-LOC', 'train')
        
        train_data = datasets.ImageFolder(
            root=train_path,
            transform=transform
        )
        print(f"Loading ImageNet dataset from: {train_path}")
        
    elif dataset_name == 'cifar10':
        # CIFAR10 configuration
        transform = transforms.Compose([
            transforms.Resize(args.image_size, antialias=True),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),                                # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5]*3)        # [-1,1]
        ])
        
        train_data = datasets.CIFAR10(
            root=args.dataset_path, 
            train=True, 
            transform=transform, 
            download=True
        )
        print(f"Loading CIFAR10 dataset")
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: 'imagenet', 'cifar10'")
    
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True
    )
    
    return train_loader

def weights_init(m):
    classname = m.__class__.__name__
    # Only initialize basic Conv layers that have weights, not composite modules
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def denormalize_tensor(tensor):
    """Convert tensor from [-1, 1] to [0, 1] range for proper display/saving"""
    return (tensor + 1.0) / 2.0

def normalize_tensor(tensor):
    """Convert tensor from [0, 1] to [-1, 1] range"""
    return tensor * 2.0 - 1.0


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(denormalize_tensor(x).cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[0].set_title('Input')
    axarr[1].imshow(denormalize_tensor(reconstruction).cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].set_title('Reconstruction')
    axarr[2].imshow(denormalize_tensor(half_sample).cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].set_title('Half Sample')
    axarr[3].imshow(denormalize_tensor(full_sample).cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].set_title('Full Sample')
    
    # Remove axis ticks
    for ax in axarr:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# =========================
# ---- Flexible Loader ----
# =========================
def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _ws_from_table_len(L: int) -> int:
    s = int(round(math.sqrt(L)))
    return (s + 1) // 2

@torch.no_grad()
def _resize_rel_pos_bias_table(param: torch.Tensor, new_len: int, n_heads: int):
    old_len, nH_ckpt = param.shape
    if nH_ckpt != n_heads:
        return None
    old_ws = _ws_from_table_len(old_len)
    new_ws = _ws_from_table_len(new_len)
    if old_ws == new_ws:
        return param
    old_sz = 2 * old_ws - 1
    new_sz = 2 * new_ws - 1
    t = param.view(old_sz, old_sz, n_heads).permute(2, 0, 1).unsqueeze(1)  # (nH,1,old,old)
    t = F.interpolate(t, size=(new_sz, new_sz), mode="bicubic", align_corners=False)
    t = t.squeeze(1).permute(1, 2, 0).contiguous().view(new_sz * new_sz, n_heads)
    return t

def _find_stage_depths(sd: dict, pattern: str):
    depths = {}
    rgx = re.compile(pattern)
    for k in sd.keys():
        m = rgx.match(k)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            depths[i] = max(depths.get(i, 0), j + 1)
    return depths

def _depth_surgery_map(src_sd: dict, dst_sd: dict):
    out = {}

    enc_src_pat = r'^encoder\.stages\.(\d+)\.blocks\.(\d+)\.(.*)'
    dec_src_pat = r'^decoder\.upsample_blocks\.(\d+)\.swin_blocks\.(\d+)\.(.*)'
    enc_dst_pat = enc_src_pat
    dec_dst_pat = dec_src_pat

    rgx_enc_dst = re.compile(enc_dst_pat)
    rgx_dec_dst = re.compile(dec_dst_pat)

    old_enc_depths = _find_stage_depths(src_sd, enc_src_pat)
    old_dec_depths = _find_stage_depths(src_sd, dec_src_pat)
    new_enc_depths = _find_stage_depths(dst_sd, enc_dst_pat)
    new_dec_depths = _find_stage_depths(dst_sd, dec_dst_pat)

    # 1) 기본: shape 일치 키는 그대로
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            out[k] = v

    # 2) Encoder: 새 블록 복사 초기화
    for k in list(dst_sd.keys()):
        m = rgx_enc_dst.match(k)
        if not m:
            continue
        i, j, tail = int(m.group(1)), int(m.group(2)), m.group(3)
        od = old_enc_depths.get(i, 0)
        nd = new_enc_depths.get(i, 0)
        if j >= nd:
            continue
        if k in out:
            continue
        if j < od:
            src_k = f"encoder.stages.{i}.blocks.{j}.{tail}"
            if src_k in src_sd and dst_sd[k].shape == src_sd[src_k].shape:
                out[k] = src_sd[src_k]
        elif od > 0:
            # 새 블록: 마지막 기존 블록 복사
            src_k = f"encoder.stages.{i}.blocks.{od-1}.{tail}"
            if src_k in src_sd and dst_sd[k].shape == src_sd[src_k].shape:
                out[k] = src_sd[src_k]

    # 3) Decoder: 새 블록 복사 초기화
    for k in list(dst_sd.keys()):
        m = rgx_dec_dst.match(k)
        if not m:
            continue
        i, j, tail = int(m.group(1)), int(m.group(2)), m.group(3)
        od = old_dec_depths.get(i, 0)
        nd = new_dec_depths.get(i, 0)
        if j >= nd:
            continue
        if k in out:
            continue
        if j < od:
            src_k = f"decoder.upsample_blocks.{i}.swin_blocks.{j}.{tail}"
            if src_k in src_sd and dst_sd[k].shape == src_sd[src_k].shape:
                out[k] = src_sd[src_k]
        elif od > 0:
            src_k = f"decoder.upsample_blocks.{i}.swin_blocks.{od-1}.{tail}"
            if src_k in src_sd and dst_sd[k].shape == src_sd[src_k].shape:
                out[k] = src_sd[src_k]

    return out

def flexible_load_pretrained(model: torch.nn.Module, ckpt_path: str,
                             do_depth_surgery: bool = True,
                             interp_rpb: bool = True,
                             device: str = "cpu",
                             verbose: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "ema_state_dict", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    src_sd = _strip_module_prefix(ckpt)

    dst_sd = model.state_dict()
    mapped = {}

    # depth surgery
    if do_depth_surgery:
        mapped = _depth_surgery_map(src_sd, dst_sd)
    else:
        mapped = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}

    # relative_position_bias_table 보간
    if interp_rpb:
        for k in list(dst_sd.keys()):
            if not k.endswith("relative_position_bias_table"):
                continue
            if k in mapped and mapped[k].shape == dst_sd[k].shape:
                continue
            if k in src_sd:
                src = src_sd[k]
            else:
                continue
            old_len, nH = src.shape
            new_len, nH2 = dst_sd[k].shape
            if nH != nH2:
                continue
            resized = _resize_rel_pos_bias_table(src, new_len, nH)
            if resized is not None and resized.shape == dst_sd[k].shape:
                mapped[k] = resized

    for k in list(dst_sd.keys()):
        if k in mapped:
            continue
        if k.endswith("patch_embed.proj.weight") and "residual_block" in k:
            weight_shape = dst_sd[k].shape
            if len(weight_shape) == 4 and weight_shape[0] == weight_shape[1]:  # (out_ch, in_ch, 1, 1)
                identity_weight = torch.zeros_like(dst_sd[k])
                for i in range(weight_shape[0]):
                    identity_weight[i, i, 0, 0] = 1.0
                mapped[k] = identity_weight
                if verbose:
                    print(f"[FlexibleLoad] Initialized {k} as identity")
        elif k.endswith("patch_embed.proj.bias") and "residual_block" in k:
            mapped[k] = torch.zeros_like(dst_sd[k])
            if verbose:
                print(f"[FlexibleLoad] Initialized {k} as zero")

    merged = {**dst_sd, **{k: v for k, v in mapped.items() if k in dst_sd and dst_sd[k].shape == v.shape}}
    incompatible = model.load_state_dict(merged, strict=False)
    if hasattr(model, "codebook") and hasattr(model.codebook, "sync_ema_buffers"):
        ema_keys = {"codebook.ema_cluster_size", "codebook.ema_w"}
        if not ema_keys.issubset(mapped.keys()):
            model.codebook.sync_ema_buffers()
            if verbose:
                print("[FlexibleLoad] Synced missing codebook EMA buffers from embedding weights")
    if verbose:
        print(f"[FlexibleLoad] loaded tensors: {len(mapped)}")
        if hasattr(incompatible, "missing_keys") and incompatible.missing_keys:
            print(f"[FlexibleLoad] missing: {len(incompatible.missing_keys)}")
        if hasattr(incompatible, "unexpected_keys") and incompatible.unexpected_keys:
            print(f"[FlexibleLoad] unexpected: {len(incompatible.unexpected_keys)}")
    loaded_names = set(mapped.keys())
    return loaded_names
