!pip install diffusers accelerate transformers peft safetensors scikit-learn pandas tqdm



CXR Diffusion Training Guide
Directory Structure
Ensure your Google Drive (/content/drive/MyDrive/CXR_Diffusion/) is organized as follows:

Plaintext
/content/drive/MyDrive/CXR_Diffusion/
├── train.py
├── dataloader.py
├── diffusion_train_single_frontal_clean.csv
└── CheXpert-v1.0-small/
    └── train/
1. Mount Google Drive
Before running the scripts in Google Colab, mount your drive to access the dataset and scripts:

Python
from google.colab import drive
drive.mount('/content/drive')

2. Start Training from Scratch
Navigate to your working directory and execute the training script. You must specify the --finetune_target to isolate your experiments (use unet for UNet-only LoRA, or unet_clip for jointly training the UNet and Text Encoder).

Bash
cd /content/drive/MyDrive/CXR_Diffusion

# Example: Training UNet + Text Encoder
python train.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --finetune_target "unet_clip"
(Note: If using multiple GPUs, replace python train.py with accelerate launch train.py).

3. Output Checkpoints
Because the script dynamically isolates experiments, it will generate a checkpoints directory containing subfolders for your specific target, keeping your saved states organized per epoch:

Plaintext
checkpoints/
├── unet/
│   ├── epoch_1/
│   ├── epoch_2/
│   └── ...
└── unet_clip/
    ├── epoch_1/
    ├── epoch_2/
    └── ...

4. Resuming Training (Colab Disconnects)
If Google Colab disconnects during training (for example, successfully saving through epoch_5 of a unet_clip run), you can resume the process without losing progress.

Option A: Resume from a specific epoch
Point the script to the exact folder of the last successful save to continue (e.g., from epoch 6 to 10):

Bash
python train.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --finetune_target "unet_clip" \
    --resume_from_checkpoint "/content/drive/MyDrive/CXR_Diffusion/checkpoints/unet_clip/epoch_5"
Option B: Resume automatically from the latest epoch
Instead of providing the full path, append latest to automatically pick up from the most recent save within your specified target directory:

Bash
    --resume_from_checkpoint latest
    
5. Loading the Model for Inference
Once training is complete, load your base model and inject the saved LoRA weights to generate new images:

Python
import torch
from diffusers import StableDiffusionPipeline

# 1. Load Base Model
pipe = StableDiffusionPipeline.from_pretrained(
    "P-RAJIV/cxr_stable_diffusion_v1_5", 
    torch_dtype=torch.float16
).to("cuda")

checkpoint_dir = "/content/drive/MyDrive/CXR_Diffusion/checkpoints/unet_clip/epoch_10"

# 2. Load UNet LoRA
pipe.load_lora_weights(checkpoint_dir, weight_name="unet_lora/adapter_model.safetensors")

# 3. Load Text Encoder LoRA (if you trained with unet_clip)
import os
if os.path.exists(os.path.join(checkpoint_dir, "text_encoder_lora")):
    pipe.load_lora_weights(checkpoint_dir, weight_name="text_encoder_lora/adapter_model.safetensors")

# 4. Generate Image
prompt = "Chest X-ray showing Pneumonia"
image = pipe(prompt).images[0]
image.save("generated_cxr.png")
Important Note: Both dataloader.py and train.py must be located in the exact same directory. The training script uses the import from dataloader import CXRDiffusionDataset, collate_fn, which requires this flat structure to function correctly.



















CXR Diffusion Training Guide (Combined Pipeline)
Directory Structure
Ensure your Google Drive (/content/drive/MyDrive/CXR_Diffusion/) is organized to support both models. If you plan to train using the PRISM model, you must download its model.safetensors file first.

Plaintext
/content/drive/MyDrive/CXR_Diffusion/
├── train_combined.py
├── dataloader.py
├── diffusion_train_single_frontal_clean.csv
├── prism_model.safetensors                # Required ONLY if using --model_type prism
└── CheXpert-v1.0-small/
    └── train/
1. Mount Google Drive
Before running the scripts in Google Colab, mount your drive:

Python
from google.colab import drive
drive.mount('/content/drive')
2. Start Training (4 Experimental Setups)
Navigate to your working directory. The script automatically creates isolated subfolders for your checkpoints based on the --model_type and --lora_mode you choose.

Bash
cd /content/drive/MyDrive/CXR_Diffusion
Experiment A: P-RAJIV Base Model (UNet Only)

Bash
python train_combined.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --model_type "prajiv" \
    --lora_mode "unet" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4
Experiment B: P-RAJIV Base Model (UNet + Text Encoder)

Bash
python train_combined.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --model_type "prajiv" \
    --lora_mode "unet_clip" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4
Experiment C: PRISM Base Model (UNet Only)
Note: Requires --prism_weights pointing to the downloaded 3.44 GB file.

Bash
python train_combined.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --model_type "prism" \
    --prism_weights "/content/drive/MyDrive/CXR_Diffusion/prism_model.safetensors" \
    --lora_mode "unet" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4
Experiment D: PRISM Base Model (UNet + Text Encoder)

Bash
python train_combined.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --model_type "prism" \
    --prism_weights "/content/drive/MyDrive/CXR_Diffusion/prism_model.safetensors" \
    --lora_mode "unet_clip" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 4
(Note: If using multiple GPUs, replace python with accelerate launch).

3. Output Checkpoints Organization
Because the script dynamically isolates experiments, it will securely organize your saved states without overlapping:

Plaintext
checkpoints/
├── prajiv/
│   ├── unet/
│   │   └── epoch_10/
│   └── unet_clip/
│       └── epoch_10/
└── prism/
    ├── unet/
    │   └── epoch_10/
    └── unet_clip/
        └── epoch_10/
4. Resuming Training (Colab Disconnects)
If Colab disconnects, you can resume exactly where you left off by appending --resume_from_checkpoint latest. You must provide the same --model_type and --lora_mode flags so the script looks in the correct folder.

Bash
python train_combined.py \
    --csv_path "/content/drive/MyDrive/CXR_Diffusion/diffusion_train_single_frontal_clean.csv" \
    --img_dir "/content/drive/MyDrive/CXR_Diffusion" \
    --output_dir "/content/drive/MyDrive/CXR_Diffusion/checkpoints" \
    --model_type "prajiv" \
    --lora_mode "unet_clip" \
    --resume_from_checkpoint latest
5. Loading the Model for Inference
Once training is complete, load your base model and inject the saved LoRA weights. Your loading logic changes slightly depending on if you used P-RAJIV or PRISM as the base.

Python
import torch
import os
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

# ==========================================
# STEP 1: LOAD THE CORRECT BASE MODEL
# ==========================================

MODEL_TYPE = "prajiv" # Change to "prism" if evaluating a PRISM run

if MODEL_TYPE == "prajiv":
    pipe = StableDiffusionPipeline.from_pretrained(
        "P-RAJIV/cxr_stable_diffusion_v1_5", 
        torch_dtype=torch.float16
    ).to("cuda")

elif MODEL_TYPE == "prism":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Load PRISM full U-Net weights into the pipeline
    prism_state_dict = load_file("/content/drive/MyDrive/CXR_Diffusion/prism_model.safetensors")
    pipe.unet.load_state_dict(prism_state_dict, strict=False)

# ==========================================
# STEP 2: LOAD LORA ADAPTERS
# ==========================================

# Point to your best epoch for the specific experiment
checkpoint_dir = "/content/drive/MyDrive/CXR_Diffusion/checkpoints/prajiv/unet_clip/epoch_10"

# 2A. Load UNet LoRA
pipe.load_lora_weights(checkpoint_dir, weight_name="unet_lora/adapter_model.safetensors")

# 2B. Load Text Encoder LoRA (if you trained using --lora_mode unet_clip)
if os.path.exists(os.path.join(checkpoint_dir, "text_encoder_lora")):
    pipe.load_lora_weights(checkpoint_dir, weight_name="text_encoder_lora/adapter_model.safetensors")

# ==========================================
# STEP 3: GENERATE
# ==========================================
prompt = "Chest X-ray showing Pneumonia"
image = pipe(prompt).images[0]
image.save("generated_cxr.png")

















How to Execute
To keep your real and synthetic checkpoints completely separate (so they don't overwrite each other's log.json files), you just change the --output_dir name when you run the script.

1. Run the Real Data Baseline

Bash
python train_classifier.py \
    --train_csv "CheXpert-v1.0-small/classifier_train_single_frontal_clean_real1500.csv" \
    --test_csv "CheXpert-v1.0-small/global_test_single_frontal_clean.csv" \
    --output_dir "real_checkpoint" \
    --epochs 10 \
    --batch_size 32
2. Run the Real + Synthetic Dataset

Bash
python train_classifier.py \
    --train_csv "CheXpert-v1.0-small/classifier_train_single_frontal_clean_combined.csv" \
    --test_csv "CheXpert-v1.0-small/global_test_single_frontal_clean.csv" \
    --output_dir "real_synthetic_checkpoint" \
    --epochs 10 \
    --batch_size 32
3. Resume from a specific epoch (Example)
If Colab disconnects during Epoch 4 of your synthetic run, you can resume by pointing exactly to the last saved .pth file:

Bash
python train_classifier.py \
    --train_csv "CheXpert-v1.0-small/classifier_train_single_frontal_clean_combined.csv" \
    --test_csv "CheXpert-v1.0-small/global_test_single_frontal_clean.csv" \
    --output_dir "real_synthetic_checkpoint" \
    --resume_from "real_synthetic_checkpoint/epoch_4.pth" \
    --epochs 10 \
    --batch_size 32
This will automatically load the model weights, restore the optimizer state, read the existing log.json to keep your history intact, and begin training at Epoch 5.