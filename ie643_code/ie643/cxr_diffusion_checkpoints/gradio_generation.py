import gradio as gr
import torch
import gc
import os
import random
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from peft import PeftModel

# ==========================================
# --- CONFIGURATION ---
# Update these paths to point to your files!
# ==========================================
PRAJIV_LORA_DIR = "./prajiv/unet_clip/epoch_2"
PRISM_LORA_DIR = "./prism/unet_clip/epoch_8"
PRISM_SAFETENSORS_PATH = "./prism_model.safetensors"

# Global variables to manage GPU memory and avoid reloading
current_loaded_model = None
pipe = None

def load_model(model_name):
    """Loads the requested model into VRAM, replacing the old one if necessary."""
    global current_loaded_model, pipe
    
    # If the requested model is already loaded, skip reloading!
    if current_loaded_model == model_name and pipe is not None:
        return pipe

    print(f"\nSwitching model... Cleaning up VRAM.")
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.device.type == "cuda" else torch.float32
    
    print(f"Loading {model_name} base pipeline...")
    if model_name == "Prajiv":
        pipe = StableDiffusionPipeline.from_pretrained(
            "P-RAJIV/cxr_stable_diffusion_v1_5", 
            torch_dtype=dtype
        )
        lora_dir = PRAJIV_LORA_DIR
        
    elif model_name == "Prism":
        if not os.path.exists(PRISM_SAFETENSORS_PATH):
            raise gr.Error(f"PRISM weights not found at {PRISM_SAFETENSORS_PATH}. Please update configuration.")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=dtype
        )
        print("Injecting PRISM Safetensors...")
        prism_state_dict = load_file(PRISM_SAFETENSORS_PATH)
        pipe.unet.load_state_dict(prism_state_dict, strict=False)
        lora_dir = PRISM_LORA_DIR

    # Load LoRA Weights (assuming unet_clip based on your previous logs)
    if not os.path.exists(lora_dir):
        raise gr.Error(f"LoRA directory not found at {lora_dir}. Please update configuration.")
        
    print(f"Loading LoRA adapters from {lora_dir}...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(lora_dir, "unet_lora"))
    
    text_encoder_path = os.path.join(lora_dir, "text_encoder_lora")
    if os.path.exists(text_encoder_path):
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_path)

    pipe.to(device)
    pipe.safety_checker = None # Disable to prevent black images
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True) # Hides inner progress bar

    current_loaded_model = model_name
    print(f"{model_name} successfully loaded into VRAM!")
    return pipe

def generate_image(model_choice, pathology, guidance, seed_input):
    """Gradio inference function."""
    try:
        # Ensure the correct model is loaded
        current_pipe = load_model(model_choice)
        
        # Handle Random Seed
        if seed_input <= -1:
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = int(seed_input)
            
        generator = torch.Generator(device=current_pipe.device).manual_seed(actual_seed)
        prompt = f"Chest X-ray showing {pathology}"
        
        print(f"\nGenerating: '{prompt}' | Model: {model_choice} | Seed: {actual_seed} | Guidance: {guidance}")
        
        # Generate
        image = current_pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        # Free up cache after generation
        torch.cuda.empty_cache()
        
        return image, f"Seed used: {actual_seed}"
        
    except Exception as e:
        raise gr.Error(str(e))

# ==========================================
# --- GRADIO UI LAYOUT ---
# ==========================================
with gr.Blocks(title="Synthetic CXR Generator", theme=gr.themes.Soft()) as interface:
    gr.Markdown(
        """
        # 🩻 Synthetic Chest X-Ray Generator
        Select your model, choose a pathology, and tweak your parameters. 
        """
    )
    
    with gr.Row():
        # Left Column: Controls
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=["Prajiv", "Prism"], 
                value="Prajiv", 
                label="1. Select Base Model"
            )
            
            pathology_dropdown = gr.Dropdown(
                choices=["Lung Lesion", "Pleural Other", "Pneumonia", "Enlarged Cardiomediastinum", "Fracture"], 
                value="Lung Lesion", 
                label="2. Select Target Pathology"
            )
            
            guidance_slider = gr.Slider(
                minimum=1.0, maximum=15.0, value=7.5, step=0.5, 
                label="3. Guidance Scale (CFG)",
                info="Higher values force the model to stick closer to the prompt."
            )
            
            seed_number = gr.Number(
                value=-1, 
                label="4. Random Seed", 
                precision=0,
                info="Set to -1 for a random image every time. Use a specific number to recreate an image."
            )
            
            generate_btn = gr.Button("Generate X-Ray", variant="primary", size="lg")
            
        # Right Column: Image Output
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Output", type="pil", interactive=False)
            output_seed = gr.Textbox(label="Generation Info", interactive=False)

    # Wire the button to the function
    generate_btn.click(
        fn=generate_image,
        inputs=[model_dropdown, pathology_dropdown, guidance_slider, seed_number],
        outputs=[output_image, output_seed]
    )

# if __name__ == "__main__":
#     # Launch the app. 
#     # Use share=True if you are on Colab and need a public link.
#     interface.launch(share=True, debug=True)
if __name__ == "__main__":
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )