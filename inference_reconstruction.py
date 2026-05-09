import os
import argparse
import torch
import random
from PIL import Image
from torchvision import utils as vutils
from torchvision import transforms
from efficient_vqgan import EfficientVQGAN
from utils import flexible_load_pretrained

def load_image(image_path, image_size=256):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] 범위로 정규화
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def save_reconstruction(original, reconstructed, save_path):
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    batch_size = original.size(0)
    comparison = torch.cat([original, reconstructed], dim=0)
    vutils.save_image(comparison, save_path, nrow=batch_size, padding=2, normalize=False)
    print(f"Saved reconstruction to {save_path}")


def reconstruct_single_image(model, image_path, save_path, device, image_size=256):
    image = load_image(image_path, image_size=image_size)

    image = image.to(device)
    with torch.no_grad():
        reconstructed, indices, _ = model(image)
    save_reconstruction(image.cpu(), reconstructed.cpu(), save_path)
    unique_indices = torch.unique(indices)
    print(f"Used {len(unique_indices)} / {model.codebook.K} codebook entries")
    return reconstructed


def reconstruct_from_directory(model, input_dir, output_dir, device, max_images=None, batch_size=4, num_iterations=None):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    if max_images is not None:
        image_files = image_files[:max_images]
    print(f"Processing {len(image_files)} images from {input_dir}")

    iteration_count = 0
    for batch_idx in range(0, len(image_files), batch_size):
        if num_iterations is not None and iteration_count >= num_iterations:
            print(f"Reached maximum iterations ({num_iterations}). Stopping.")
            break

        iteration_count += 1
        batch_files = image_files[batch_idx:batch_idx + batch_size]
        originals = []
        reconstructed_list = []
        
        for img_file in batch_files:
            input_path = os.path.join(input_dir, img_file)
            try:
                image = load_image(input_path, image_size=256)
                image = image.to(device)
                
                with torch.no_grad():
                    reconstructed, indices, _ = model(image)
                
                originals.append(image.cpu())
                reconstructed_list.append(reconstructed.cpu())
                
                unique_indices = torch.unique(indices)
                print(f"{img_file}: Used {len(unique_indices)} / {model.codebook.K} codebook entries")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        if originals:
            originals_batch = torch.cat(originals, dim=0)
            reconstructed_batch = torch.cat(reconstructed_list, dim=0)
            output_path = os.path.join(output_dir, f"batch_{batch_idx // batch_size + 1}_reconstruction.png")
            save_reconstruction(originals_batch, reconstructed_batch, output_path)


def reconstruct_random_images(model, input_dir, output_path, device, num_images=4, image_size=256):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_dir}")
    
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    print(f"Randomly selected {len(selected_files)} images from {input_dir}:")
    for f in selected_files:
        print(f"  - {f}")
    
    originals = []
    reconstructed_list = []
    
    for img_file in selected_files:
        input_path = os.path.join(input_dir, img_file)
        try:
            image = load_image(input_path, image_size=image_size)
            image = image.to(device)
            
            with torch.no_grad():
                reconstructed, indices, _ = model(image)
            
            originals.append(image.cpu())
            reconstructed_list.append(reconstructed.cpu())
            
            unique_indices = torch.unique(indices)
            print(f"{img_file}: Used {len(unique_indices)} / {model.codebook.K} codebook entries")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    if originals:
        originals_batch = torch.cat(originals, dim=0)
        reconstructed_batch = torch.cat(reconstructed_list, dim=0)
        save_reconstruction(originals_batch, reconstructed_batch, output_path)
        print(f"\nVisualization complete! Saved to {output_path}")
    else:
        print("No images were successfully processed.")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Initializing VQGAN model...")
    model = EfficientVQGAN(args).to(device)
    model.eval()
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    print(f"Loading pretrained weights from {args.checkpoint_path}")
    loaded_params = flexible_load_pretrained(
        model,
        ckpt_path=args.checkpoint_path,
        do_depth_surgery=True,
        interp_rpb=True,
        device=device,
        verbose=True
    )
    print(f"Loaded {len(loaded_params)} parameters from checkpoint")
    if args.random_sample:
        input_dir = args.input_dir or "data/imagenet"
        output_path = os.path.join(args.output_dir, "visualization_result.jpg")
        print(f"Randomly sampling {args.num_samples} images from {input_dir}")
        reconstruct_random_images(
            model, input_dir, output_path, device, 
            num_images=args.num_samples, image_size=args.image_size
        )
    elif args.input_image:
        print(f"Processing single image: {args.input_image}")
        output_path = args.output_path or "reconstruction_result.png"
        reconstruct_single_image(model, args.input_image, output_path, device, args.image_size)
    elif args.input_dir:
        output_dir = args.output_dir or "reconstruction_results"
        reconstruct_from_directory(
            model, args.input_dir, output_dir, device, args.max_images, 
            args.batch_size, args.num_iterations
        )
    else:
        print("Please specify either --input_image, --input_dir, or --random_sample")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN Reconstruction Inference")
    parser.add_argument('--image-size', type=int, default=256, help='Image resolution')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--num-codebook-vectors', type=int, default=4096, help='Number of codebook vectors')
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss weight')
    parser.add_argument('--encoder-stem', type=str, default='hybrid', choices=['hybrid', 'patch4'],
                        help='Encoder input stem')
    parser.add_argument('--encoder-pre-quant-blocks', type=int, default=1,
                        help='Residual conv blocks before quant_conv')
    parser.add_argument('--decoder-min-upsample-channels', type=int, default=64,
                        help='Minimum channel count kept after decoder PatchExpanding stages')
    parser.add_argument('--decoder-refine-blocks', type=int, default=2,
                        help='Number of full-resolution decoder refinement blocks')
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
                        help='Chunk size for nearest-code lookup')
    parser.add_argument('--dead-code-threshold', type=float, default=0.1,
                        help='EMA cluster size below which a code is considered dead')
    parser.add_argument('--fused-window-process', action='store_true', default=True,
                        help='Use native fused Swin window processing when available')
    parser.add_argument('--checkpoint-path', type=str, default="checkpoints/conv3_last.pt",
                        help='Path to pretrained VQGAN checkpoint')
    parser.add_argument('--input-image', type=str, default=None,
                        help='Path to single input image')
    parser.add_argument('--input-dir', type=str, default="data/archive/arctic_fox",
                        help='Path to directory containing input images')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output path for single image reconstruction')
    parser.add_argument('--output-dir', type=str, default='reconstruction_results',
                        help='Output directory for batch reconstruction')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for batch mode)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of images to display in one row (default: 4)')
    parser.add_argument('--num-iterations', type=int, default=3,
                        help='Number of iterations (batches) to process (default: None, process all)')
    parser.add_argument('--random-sample', action='store_true',
                        help='Randomly sample images from input directory')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of random images to sample (default: 4)')
    args = parser.parse_args()
    main(args)
