import argparse
import os
import torch
import math
from torchvision.utils import save_image
from tqdm import tqdm
from pipeline import InferencePipeline

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Image Generation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--config", type=str, default="configs/config_sd.yaml", help="Path to the config file")
    parser.add_argument("--num_images_per_class", type=int, default=100, help="Number of images to generate per class (0-9)")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale (default: use config)")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps (default: use config)")
    parser.add_argument("--method", type=str, default=None, choices=['euler', 'midpoint'], help="Solver method")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Initializing pipeline with config: {args.config}")
    print(f"Using checkpoint: {args.checkpoint}")
    
    try:
        pipe = InferencePipeline(
            config_path=args.config,
            checkpoint_path=args.checkpoint
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    classes = [str(i) for i in range(10)]
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Generating {args.num_images_per_class} images for each class (Total: {args.num_images_per_class * 10})")

    for class_name in classes:
        class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        generated_count = 0
        pbar = tqdm(total=args.num_images_per_class, desc=f"Class {class_name}")
        
        while generated_count < args.num_images_per_class:
            current_batch_size = min(args.batch_size, args.num_images_per_class - generated_count)
            prompts = [class_name] * current_batch_size
            images = pipe(
                prompts, 
                guidance_scale=args.guidance_scale,
                steps=args.steps,
                method=args.method
            )
            for i, image in enumerate(images):
                save_idx = generated_count + i
                save_path = os.path.join(class_dir, f"{save_idx:05d}.png")
                
                save_image(
                    image, 
                    save_path, 
                    normalize=True, 
                    value_range=(-1, 1) 
                )
            
            generated_count += current_batch_size
            pbar.update(current_batch_size)
        
        pbar.close()

    print("\nGeneration complete!")
    print(f"Images saved to {args.output_dir}")

if __name__ == "__main__":
    main()