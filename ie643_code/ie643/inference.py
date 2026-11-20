import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from peft import PeftModel
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LoRA-tuned CXR Stable Diffusion")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, choices=["prajiv", "prism"], required=True, 
                        help="The base model used during training.")
    parser.add_argument("--lora_mode", type=str, choices=["unet", "unet_clip"], required=True, 
                        help="Whether LoRA was applied to just UNet or UNet + CLIP.")
    
    # Paths
    parser.add_argument("--model_id", type=str, default="P-RAJIV/cxr_stable_diffusion_v1_5", 
                        help="Hugging Face repo ID for base model (if prajiv).")
    parser.add_argument("--prism_weights", type=str, default=None, 
                        help="Path to PRISM model.safetensors (required if model_type is prism).")
    parser.add_argument("--epoch_dir", type=str, required=True, 
                        help="Path to the saved epoch directory (e.g., cxr_diffusion_checkpoints/prajiv/unet/epoch_10).")
    parser.add_argument("--output_dir", type=str, default="inference_results", 
                        help="Directory to save the generated images.")
    
    # Generation Settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()

def load_base_pipeline(args, device):
    """Loads the base Stable Diffusion pipeline before applying LoRA."""
    
    print(f"\nLoading base {args.model_type.upper()} pipeline...")
    
    if args.model_type == "prajiv":
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16
        )
        
    elif args.model_type == "prism":
        if args.prism_weights is None:
            raise ValueError("For PRISM inference, you must provide --prism_weights")
            
        base_model = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16
        )
        
        # Inject PRISM UNet weights
        print(f"Injecting PRISM weights from {args.prism_weights}")
        prism_state_dict = load_file(args.prism_weights)
        pipe.unet.load_state_dict(prism_state_dict, strict=False)

    pipe = pipe.to(device)
    return pipe

def apply_lora_adapters(pipe, args):
    """Loads and attaches the LoRA weights saved during training using PEFT."""
    
    print(f"\nLoading LoRA adapters from: {args.epoch_dir}")
    
    unet_lora_path = os.path.join(args.epoch_dir, "unet_lora")
    if not os.path.exists(unet_lora_path):
        raise FileNotFoundError(f"U-Net LoRA directory not found at {unet_lora_path}")
    
    # 1. Wrap the base UNet with the saved PEFT adapter
    print("-> Loading U-Net LoRA...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)
    
    # 2. Wrap the Text Encoder if applicable
    if args.lora_mode == "unet_clip":
        te_lora_path = os.path.join(args.epoch_dir, "text_encoder_lora")
        if not os.path.exists(te_lora_path):
            raise FileNotFoundError(f"Text Encoder LoRA directory not found at {te_lora_path}")
        
        print("-> Loading Text Encoder LoRA...")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_lora_path)

    # Ensure the models are correctly placed on the target device and dtype
    pipe.to(pipe.device)

    return pipe

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Load Pipeline & LoRA
    pipe = load_base_pipeline(args, device)
    pipe = apply_lora_adapters(pipe, args)
    
    # Optional: Enable memory optimizations
    # pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # 3. Define the Classes (Target 5 + Normal) based on your DataLoader
    target_classes = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Lung Lesion",
        "Pneumonia",
        "Pleural Other",
        "Fracture"
    ]
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    print("\n" + "="*50)
    print("STARTING INFERENCE")
    print("="*50)

    # List to store images for the grid
    generated_images = []

    # 4. Generate an image for each class
    for cls in target_classes:
        # Replicate the caption logic from your dataloader
        if cls == "No Finding":
            prompt = "Normal chest X-ray with no significant findings"
            filename_suffix = "normal"
        else:
            prompt = f"Chest X-ray showing {cls}"
            filename_suffix = cls.lower().replace(" ", "_")
            
        print(f"Generating: {cls}")
        print(f"Prompt: '{prompt}'")
        
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        
        # Save image
        run_name = f"{args.model_type}_{args.lora_mode}"
        filename = f"{run_name}_{filename_suffix}.png"
        save_path = os.path.join(args.output_dir, filename)
        image.save(save_path)
        print(f"Saved to {save_path}\n")
        # Store for the grid
        generated_images.append((cls, image))


    # ==========================================
    # 5. CREATE AND SHOW GRID
    # ==========================================
    print("Generating 2x3 grid...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Results: {args.model_type.upper()} ({args.lora_mode.upper()})", fontsize=18, fontweight='bold')
    
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(generated_images):
            cls_name, img = generated_images[idx]
            ax.imshow(img)
            ax.set_title(cls_name, fontsize=12)
            ax.axis('off')
            
    plt.tight_layout()
    
    # Save the grid image
    run_name = f"{args.model_type}_{args.lora_mode}"
    grid_path = os.path.join(args.output_dir, f"{run_name}_all_classes_grid.png")
    plt.savefig(grid_path, bbox_inches='tight')
    print(f"Saved grid to: {grid_path}")
    
    # Show the grid on screen
    plt.show()

if __name__ == "__main__":
    main()