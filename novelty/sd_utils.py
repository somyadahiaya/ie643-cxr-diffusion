"""Loading P-Rajiv / PRISM + LoRA, and a deterministic batched DDIM sampler (shared by train / generate / eval)."""
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from peft import PeftModel, LoraConfig, get_peft_model

import os
# Local copy of P-RAJIV/cxr_stable_diffusion_v1_5: VAE/text-encoder/tokenizer/scheduler from the HF hub, UNet base
# weights recovered from the team's prajiv/unet_clip/epoch_2 accelerate checkpoint (frozen base, fp16).
_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prajiv_base")
PRAJIV_ID = _LOCAL if os.path.isdir(_LOCAL) else "P-RAJIV/cxr_stable_diffusion_v1_5"
SD15_ID = "runwayml/stable-diffusion-v1-5"


def load_sd(model_type, prism_weights=None, dtype=torch.bfloat16, device="cuda"):
    if model_type == "prajiv":
        pipe = StableDiffusionPipeline.from_pretrained(PRAJIV_ID, torch_dtype=dtype, safety_checker=None,
                                                       requires_safety_checker=False)
    else:
        from safetensors.torch import load_file
        pipe = StableDiffusionPipeline.from_pretrained(SD15_ID, torch_dtype=dtype, safety_checker=None,
                                                       requires_safety_checker=False)
        missing, unexpected = pipe.unet.load_state_dict(load_file(prism_weights), strict=False)
        assert not missing and not unexpected, "PRISM weights do not match SD1.5 UNet"
        pipe.unet.to(dtype)
    for m in (pipe.unet, pipe.text_encoder, pipe.vae):
        m.requires_grad_(False)
    pipe.to(device)
    ddim = DDIMScheduler.from_config(pipe.scheduler.config)
    ddpm = DDPMScheduler.from_config(pipe.scheduler.config)
    return pipe, ddim, ddpm


def attach_lora(pipe, lora_dir=None, lora_mode="unet_clip", trainable=False, rank=16):
    """Load existing LoRA adapters (lora_dir/unet_lora, lora_dir/text_encoder_lora) or create fresh ones."""
    import os
    if lora_dir:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.join(lora_dir, "unet_lora"), is_trainable=trainable)
        te_dir = os.path.join(lora_dir, "text_encoder_lora")
        if lora_mode == "unet_clip" and os.path.isdir(te_dir):
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_dir, is_trainable=trainable)
    elif trainable:
        pipe.unet = get_peft_model(pipe.unet, LoraConfig(r=rank, lora_alpha=2 * rank, init_lora_weights="gaussian",
                                                         target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        if lora_mode == "unet_clip":
            pipe.text_encoder = get_peft_model(pipe.text_encoder, LoraConfig(
                r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    # trainable LoRA params in fp32 (avoids fp16/bf16 optimizer issues)
    for m in (pipe.unet, pipe.text_encoder):
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
                p.requires_grad_(trainable)
    return pipe


def set_lora_scale(pipe, s):
    """Interpolate between the base model (s=0) and the full LoRA (s=1) at inference (DRaFT-style weight scaling)."""
    n = 0
    for net in (pipe.unet, pipe.text_encoder):
        for m in net.modules():
            if hasattr(m, "scaling") and isinstance(m.scaling, dict) and m.scaling:
                if not hasattr(m, "_orig_scaling"):
                    m._orig_scaling = dict(m.scaling)
                for k in m.scaling:
                    m.scaling[k] = m._orig_scaling[k] * s
                n += 1
    return n


def encode_prompts(pipe, prompts, device="cuda"):
    ids = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(device)
    return pipe.text_encoder(ids)[0]


def init_latents(seeds, device="cuda", dtype=torch.float32):
    """One CPU generator per image -> identical noise for a given seed regardless of batching."""
    return torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
                       ).to(device, dtype)


def cfg_eps(unet, z, t, cond, uncond, scale):
    out = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([uncond, cond])).sample
    e_u, e_c = out.chunk(2)
    return e_u + scale * (e_c - e_u)


@torch.no_grad()
def sample(pipe, ddim, prompts, seeds, steps=50, cfg=7.5, device="cuda", return_tensor=False):
    """Returns PIL images (or a [-1,1] tensor) for prompts with per-image seeds."""
    dtype = pipe.unet.dtype if hasattr(pipe.unet, "dtype") else torch.bfloat16
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cond = encode_prompts(pipe, prompts, device)
        uncond = encode_prompts(pipe, [""] * len(prompts), device)
        z = init_latents(seeds, device) * ddim.init_noise_sigma
        ddim.set_timesteps(steps, device=device)
        for t in ddim.timesteps:
            eps = cfg_eps(pipe.unet, z, t, cond, uncond, cfg)
            z = ddim.step(eps.float(), t, z.float()).prev_sample
        z = z / pipe.vae.config.scaling_factor
        img = torch.cat([pipe.vae.decode(z[i:i + 4]).sample.float().clamp(-1, 1)   # chunked: VAE decode is the
                         for i in range(0, len(z), 4)])                            # memory peak at 512px
    if return_tensor:
        return img
    from PIL import Image
    arr = ((img.mean(1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(a, mode="L") for a in arr]
