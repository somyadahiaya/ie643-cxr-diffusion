import os
import argparse
import torch
import math
import torch.nn.functional as F

from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler # ADD THIS LINE
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.utils.data import WeightedRandomSampler, DataLoader

from dataloader import CXRDiffusionDataset, collate_fn


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default="")

    # Model
    parser.add_argument("--model_type", type=str, choices=["prajiv", "prism"], default="prajiv")
    parser.add_argument("--model_id", type=str, default="P-RAJIV/cxr_stable_diffusion_v1_5")
    parser.add_argument("--prism_weights", type=str, default=None, help="Path to PRISM model.safetensors")

    # LoRA
    parser.add_argument("--lora_mode", type=str, choices=["unet", "unet_clip"], default="unet")
    parser.add_argument("--lora_rank", type=int, default=8)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", help='Choices: "linear", "cosine", "constant_with_warmup"')
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    # Checkpoints
    parser.add_argument("--output_dir", type=str, default="cxr_diffusion_checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint folder or 'latest'")

    return parser.parse_args()


# ============================================================
# LOAD BASE MODEL
# ============================================================

def load_model(args):
    # CASE 1: P-RAJIV
    if args.model_type == "prajiv":
        print("\n" + "=" * 70)
        print("Loading P-RAJIV CXR Stable Diffusion")
        print("=" * 70)

        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16
        )

        # Override the inference scheduler with DDPM for training
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        return pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet, noise_scheduler

    # CASE 2: PRISM
    elif args.model_type == "prism":
        print("\n" + "=" * 70)
        print("Loading PRISM")
        print("=" * 70)

        if args.prism_weights is None:
            raise ValueError("For PRISM you must provide --prism_weights")

        base_model = "runwayml/stable-diffusion-v1-5"
        print(f"Base model: {base_model}")

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16
        )

        print(f"\nLoading PRISM weights:\n{args.prism_weights}")
        prism_state_dict = load_file(args.prism_weights)
        missing_keys, unexpected_keys = pipe.unet.load_state_dict(prism_state_dict, strict=False)
        print("\nPRISM weights loaded.")

        if len(missing_keys) > 0:
            print(f"Missing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "PRISM weights did not fully match the SD 1.5 UNet."
            )
        print("=" * 70)

        # Override the inference scheduler with DDPM for training
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        return pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet, noise_scheduler


# ============================================================
# APPLY LoRA
# ============================================================

def apply_lora(unet, text_encoder, lora_mode, lora_rank):
    # Freeze original U-Net
    unet.requires_grad_(False)

    # U-Net LoRA
    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,  # Doubles update strength
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    # unet_lora_config = LoraConfig(
    #                                 r=lora_rank,
    #                                 lora_alpha=lora_rank * 2,  # Doubles update strength
    #                                 init_lora_weights="gaussian",
    #                                 target_modules=[
    #                                     "to_k", "to_q", "to_v", "to_out.0",
    #                                     "ff.net.0.proj", "ff.net.2"  # <-- ADD THIS: Feed-Forward layers
    #                                 ]
    #                             )
    unet = get_peft_model(unet, unet_lora_config)
    print("\nU-Net LoRA Trainable Parameters:")
    unet.print_trainable_parameters()

    # Optional CLIP LoRA
    if lora_mode == "unet_clip":
        text_encoder.requires_grad_(False)
        text_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        text_encoder = get_peft_model(text_encoder, text_lora_config)
        print("\nText Encoder LoRA Trainable Parameters:")
        text_encoder.print_trainable_parameters()

    return unet, text_encoder


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def get_checkpoint_path(args):
    if args.resume_from_checkpoint is None:
        return None

    if args.resume_from_checkpoint == "latest":
        if not os.path.exists(args.output_dir):
            return None
        checkpoints = []
        for name in os.listdir(args.output_dir):
            if name.startswith("epoch_"):
                try:
                    epoch_number = int(name.split("_")[1])
                    checkpoints.append((epoch_number, name))
                except ValueError:
                    pass
        if len(checkpoints) == 0:
            return None
        checkpoints.sort()
        latest_name = checkpoints[-1][1]
        return os.path.join(args.output_dir, latest_name)

    return args.resume_from_checkpoint


def get_balanced_dataloader(dataset, batch_size, num_workers, collate_fn):
    # 1. Extract the active class name for each row
    labels = []
    for _, row in dataset.df.iterrows():
        for col in dataset.target_classes:
            if row[col] == 1.0:
                labels.append(col)
                break

    # 2. Calculate class frequencies & inverse weights
    from collections import Counter
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # 3. Assign weight to each dataset sample
    sample_weights = [class_weights[label] for label in labels]

    # 4. Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataset),  # Total samples drawn per epoch
        replacement=True
    )

    # 5. Return DataLoader (Note: shuffle MUST be omitted/False when sampler is used)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # --------------------------------------------------------
    # 1. ISOLATE OUTPUT DIRECTORY
    # Automatically prevents overlapping experiments
    # e.g. cxr_diffusion_checkpoints/prism/unet_clip
    # --------------------------------------------------------
    args.output_dir = os.path.join(args.output_dir, args.model_type, args.lora_mode)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    os.makedirs(args.output_dir, exist_ok=True)
    device = accelerator.device
    print("\nDevice:", device)

    # --------------------------------------------------------
    # 2. LOAD MODEL & APPLY LoRA
    # --------------------------------------------------------
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_model(args)
    unet, text_encoder = apply_lora(unet, text_encoder, args.lora_mode, args.lora_rank)

    # --------------------------------------------------------
    # 3. FREEZE COMPONENTS & ASSIGN DEVICES
    # --------------------------------------------------------
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device=device, dtype=torch.float16)

    if args.lora_mode == "unet":
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        text_encoder = text_encoder.to(device=device, dtype=torch.float16)

    # --------------------------------------------------------
    # 4. OPTIMIZER & DATASET
    # --------------------------------------------------------
    trainable_parameters = [p for p in unet.parameters() if p.requires_grad]
    if args.lora_mode == "unet_clip":
        trainable_parameters += [p for p in text_encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    dataset = CXRDiffusionDataset(csv_path=args.csv_path, base_image_dir=args.img_dir, resolution=512)
    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    print("Number of images:", len(dataset))

    # dataloader = DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    # )
    dataloader = get_balanced_dataloader(
                                            dataset=dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn
                                        )

    # --- ADD THIS BLOCK ---
    # total_training_steps = len(dataloader) * args.epochs
    
    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=total_training_steps * args.gradient_accumulation_steps,
    # )

    # Calculate the ACTUAL number of optimizer update steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = num_update_steps_per_epoch * args.epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    # ----------------------

    # --------------------------------------------------------
    # 5. PREPARE ACCELERATOR DYNAMICALLY
    # --------------------------------------------------------
    if args.lora_mode == "unet_clip":
        unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, dataloader, lr_scheduler
        )
    else:
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

    # --------------------------------------------------------
    # 6. RESUME FROM CHECKPOINT
    # --------------------------------------------------------
    start_epoch = 0
    checkpoint_path = get_checkpoint_path(args)

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print("\n" + "=" * 70)
            print("RESUMING TRAINING")
            print("=" * 70)
            print("Checkpoint:", checkpoint_path)
            
            accelerator.load_state(checkpoint_path)
            folder_name = os.path.basename(checkpoint_path)
            if folder_name.startswith("epoch_"):
                start_epoch = int(folder_name.split("_")[1])
            print("Starting from epoch:", start_epoch + 1)
        else:
            print(f"\nCheckpoint not found: {checkpoint_path}")
            print("Starting training from scratch.")

    # --------------------------------------------------------
    # 7. TRAINING LOOP
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        if args.lora_mode == "unet_clip":
            text_encoder.train()

        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

        total_loss = 0.0

        for step, batch in enumerate(progress_bar):
            
            # Dynamically handle gradient accumulation context
            if args.lora_mode == "unet_clip":
                accumulate_ctx = accelerator.accumulate(unet, text_encoder)
            else:
                accumulate_ctx = accelerator.accumulate(unet)

            with accumulate_ctx:
                pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float16)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                text_inputs = tokenizer(
                    batch["prompts"], padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                input_ids = text_inputs.input_ids.to(device)

                # Dynamically enable gradients ONLY if training unet_clip
                with torch.set_grad_enabled(args.lora_mode == "unet_clip"):
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters, 1.0)

                optimizer.step()
                lr_scheduler.step() # ADD THIS LINE
                optimizer.zero_grad()

                total_loss += loss.detach().item()
            progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")

        # --------------------------------------------------------
        # 8. SAVE CHECKPOINTS
        # --------------------------------------------------------
        avg_loss = total_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.6f}")

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"\nSaving checkpoint: {checkpoint_dir}")

        accelerator.wait_for_everyone()

        # Save optimizer + training state (Heavy)
        accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch + 1}"))

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            # Save U-Net LoRA Adapter
            unwrapped_unet = accelerator.unwrap_model(unet)
            lora_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}", "unet_lora")
            os.makedirs(lora_dir, exist_ok=True)
            unwrapped_unet.save_pretrained(lora_dir)
            print(f"U-Net LoRA adapter saved: {lora_dir}")

            # Save Text Encoder LoRA Adapter
            if args.lora_mode == "unet_clip":
                unwrapped_te = accelerator.unwrap_model(text_encoder)
                te_lora_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}", "text_encoder_lora")
                os.makedirs(te_lora_dir, exist_ok=True)
                unwrapped_te.save_pretrained(te_lora_dir)
                print(f"Text Encoder LoRA adapter saved: {te_lora_dir}")

            print("=" * 70)

    # ========================================================
    # FINISHED
    # ========================================================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print("Checkpoints saved in:")
        print(os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()