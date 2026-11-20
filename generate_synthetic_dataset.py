import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic CXR Dataset with LoRA-tuned Stable Diffusion")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, choices=["prajiv", "prism"], required=True)
    parser.add_argument("--lora_mode", type=str, choices=["unet", "unet_clip"], required=True)
    
    # Paths
    parser.add_argument("--model_id", type=str, default="P-RAJIV/cxr_stable_diffusion_v1_5")
    parser.add_argument("--prism_weights", type=str, default=None)
    parser.add_argument("--epoch_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="synthetic_dataset", 
                        help="Root directory to save the generated dataset")
    
    # Generation Settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed. Will be incremented per image.")

    return parser.parse_args()

def load_base_pipeline(args, device):
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
        pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
        
        print(f"Injecting PRISM weights from {args.prism_weights}")
        prism_state_dict = load_file(args.prism_weights)
        pipe.unet.load_state_dict(prism_state_dict, strict=False)

    pipe = pipe.to(device)
    return pipe

def apply_lora_adapters(pipe, args):
    print(f"\nLoading LoRA adapters from: {args.epoch_dir}")
    
    unet_lora_path = os.path.join(args.epoch_dir, "unet_lora")
    print("-> Loading U-Net LoRA...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_path)
    
    if args.lora_mode == "unet_clip":
        te_lora_path = os.path.join(args.epoch_dir, "text_encoder_lora")
        print("-> Loading Text Encoder LoRA...")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_lora_path)

    pipe.to(pipe.device)
    return pipe

# def main():
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. Load Pipeline & LoRA
#     pipe = load_base_pipeline(args, device)
#     pipe = apply_lora_adapters(pipe, args)
    
#     # Memory optimization (Crucial for generating thousands of images)
#     pipe.enable_model_cpu_offload()

#     # 2. Define the Target Generation Quantities
#     generation_targets = {
#         "Lung Lesion": 500,
#         "Pleural Other": 500,
#         "Pneumonia": 500,
#         "Enlarged Cardiomediastinum": 500,
#         "Fracture": 173
#     }
    
#     print("\n" + "="*50)
#     print("STARTING LARGE-SCALE SYNTHETIC GENERATION")
#     print("="*50)

#     # To keep track of generated data for a CSV file
#     csv_data = []
    
#     # 3. Generation Loop
#     for cls, num_images in generation_targets.items():
#         print(f"\nTarget: {cls} | Generating {num_images} images...")
        
#         # Create a specific subfolder for this class
#         safe_cls_name = cls.replace(" ", "_")
#         cls_output_dir = os.path.join(args.output_dir, safe_cls_name)
#         os.makedirs(cls_output_dir, exist_ok=True)
        
#         prompt = f"Chest X-ray showing {cls}"
        
#         # Use tqdm for a progress bar so you know how long it will take
#         for i in tqdm(range(num_images), desc=cls):
            
#             # THE SECRET TO DIFFERENT IMAGES: Change the seed for every single image
#             # We add a hash of the class name and the index 'i' to guarantee uniqueness
#             current_seed = args.base_seed + hash(cls) % 10000 + i
#             generator = torch.Generator(device=device).manual_seed(current_seed)
            
#             # Generate the image
#             image = pipe(
#                 prompt,
#                 num_inference_steps=args.num_inference_steps,
#                 guidance_scale=args.guidance_scale,
#                 generator=generator
#             ).images[0]
            
#             # Save image with sequential naming (e.g., Lung_Lesion_0001.png)
#             filename = f"synthetic_{safe_cls_name}_{i:04d}.png"
#             relative_path = os.path.join(safe_cls_name, filename) # Path relative to output_dir
#             absolute_path = os.path.join(cls_output_dir, filename)
            
#             image.save(absolute_path)
            
#             # Record for CSV
#             csv_data.append({
#                 "Path": relative_path,
#                 "Label": cls,
#                 "Is_Synthetic": 1
#             })
            
#         # Clear CUDA cache between classes to prevent Out-Of-Memory errors
#         torch.cuda.empty_cache()

#     # 4. Save CSV file
#     csv_path = os.path.join(args.output_dir, "synthetic_labels.csv")
#     df = pd.DataFrame(csv_data)
#     df.to_csv(csv_path, index=False)
#     print(f"\nDone! Generated {len(df)} images.")
#     print(f"Saved dataset mapping to: {csv_path}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Pipeline & LoRA
    pipe = load_base_pipeline(args, device)
    pipe = apply_lora_adapters(pipe, args)
    
    # OPTIONAL BUT RECOMMENDED: Disable the safety checker entirely for medical datasets
    # Uncomment the line below to stop the NSFW false-positives at the source
    pipe.safety_checker = None 
    
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    # 2. Define the Target Generation Quantities
    generation_targets = {
        "Lung Lesion": 500,
        "Pleural Other": 500,
        "Pneumonia": 500,
        "Enlarged Cardiomediastinum": 500,
        "Fracture": 173
    }
    
    print("\n" + "="*50)
    print("STARTING LARGE-SCALE SYNTHETIC GENERATION")
    print("="*50)

    csv_data = []
    
    # 3. Generation Loop
    for cls, num_images in generation_targets.items():
        print(f"\nTarget: {cls} | Generating {num_images} images...")
        
        safe_cls_name = cls.replace(" ", "_")
        cls_output_dir = os.path.join(args.output_dir, safe_cls_name)
        os.makedirs(cls_output_dir, exist_ok=True)
        
        prompt = f"Chest X-ray showing {cls}"
        
        for i in tqdm(range(num_images), desc=cls):
            
            valid_image = False
            seed_offset = 0 # This will increment if we get a blank image
            
            # --- THE RETRY LOOP ---
            while not valid_image:
                
                # Add seed_offset so the seed completely changes on a retry
                current_seed = args.base_seed + hash(cls) % 10000 + i + seed_offset
                generator = torch.Generator(device=device).manual_seed(current_seed)
                
                # Get the full output object, not just the image
                output = pipe(
                    prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator
                )
                
                image = output.images[0]
                
                # 1. Check if the pipeline explicitly flagged it
                is_nsfw = False
                if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected is not None:
                    is_nsfw = output.nsfw_content_detected[0]
                
                # 2. Mathematically check if the image is pure black (all pixels = 0)
                # getextrema() returns (min_pixel, max_pixel). If both are 0, it's a black box.
                extrema = image.convert("L").getextrema()
                is_blank = (extrema == (0, 0))
                
                if is_nsfw or is_blank:
                    # REJECT! Increment the offset by a large jump to guarantee a new noise pattern
                    seed_offset += 1000 
                    continue # This sends it back to the top of the 'while' loop
                
                # If we get here, the image passed both tests!
                valid_image = True
            # ----------------------
            
            # Save the valid image
            filename = f"synthetic_{safe_cls_name}_{i:04d}.png"
            relative_path = os.path.join(safe_cls_name, filename)
            absolute_path = os.path.join(cls_output_dir, filename)
            
            image.save(absolute_path)
            
            csv_data.append({
                "Path": relative_path,
                "Label": cls,
                "Is_Synthetic": 1
            })
            
        torch.cuda.empty_cache()

    # 4. Save CSV file
    csv_path = os.path.join(args.output_dir, "synthetic_labels.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"\nDone! Generated {len(df)} images.")
    print(f"Saved dataset mapping to: {csv_path}")

if __name__ == "__main__":
    main()

