import argparse
import inspect
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def load_images(data_dir: str, size: int, center_crop: bool) -> tuple[torch.Tensor, list[str]]:
    image_paths = sorted([p for p in Path(data_dir).iterdir() if p.is_file()])
    tfm = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [tfm(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(images), [p.name for p in image_paths]


def save_images(images: torch.Tensor, names: list[str], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(images, names):
        arr = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(os.path.join(out_dir, f"{prefix}_{name}"))


def encode_flux_prompt(pipe: DiffusionPipeline, prompt: str, batch_size: int, device: torch.device):
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("Loaded pipeline does not support `encode_prompt`, cannot run FLUX attack.")

    sig = inspect.signature(pipe.encode_prompt)
    kwargs = {}
    if "prompt" in sig.parameters:
        kwargs["prompt"] = [prompt] * batch_size
    if "prompt_2" in sig.parameters:
        kwargs["prompt_2"] = [prompt] * batch_size
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "num_images_per_prompt" in sig.parameters:
        kwargs["num_images_per_prompt"] = 1
    if "max_sequence_length" in sig.parameters:
        kwargs["max_sequence_length"] = 512

    encoded = pipe.encode_prompt(**kwargs)

    # FLUX pipelines typically return: (prompt_embeds, pooled_prompt_embeds, text_ids)
    # Keep this robust to slight API differences.
    if isinstance(encoded, tuple):
        if len(encoded) == 3:
            return encoded[0], encoded[1], encoded[2]
        if len(encoded) == 2:
            return encoded[0], encoded[1], None
        if len(encoded) == 1:
            return encoded[0], None, None

    raise RuntimeError("Unsupported return format from `encode_prompt`.")


def call_flux_transformer(transformer, noisy_latents, timesteps, prompt_embeds, pooled_prompt_embeds, text_ids):
    sig = inspect.signature(transformer.forward)
    kwargs = {}

    if "hidden_states" in sig.parameters:
        kwargs["hidden_states"] = noisy_latents
    elif "sample" in sig.parameters:
        kwargs["sample"] = noisy_latents
    else:
        raise RuntimeError("Unsupported transformer forward signature: missing hidden_states/sample")

    if "timestep" in sig.parameters:
        kwargs["timestep"] = timesteps
    elif "timesteps" in sig.parameters:
        kwargs["timesteps"] = timesteps

    if prompt_embeds is not None:
        if "encoder_hidden_states" in sig.parameters:
            kwargs["encoder_hidden_states"] = prompt_embeds
        elif "prompt_embeds" in sig.parameters:
            kwargs["prompt_embeds"] = prompt_embeds

    if pooled_prompt_embeds is not None:
        if "pooled_projections" in sig.parameters:
            kwargs["pooled_projections"] = pooled_prompt_embeds
        elif "pooled_prompt_embeds" in sig.parameters:
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds

    if text_ids is not None and "txt_ids" in sig.parameters:
        kwargs["txt_ids"] = text_ids

    out = transformer(**kwargs)
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    return out


def add_noise_with_fallback(scheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Add noise across scheduler variants.

    Priority:
    1) `scheduler.add_noise(...)` (DDPM-style schedulers)
    2) `scheduler.scale_noise(...)` (flow-matching schedulers)
    3) sigma-mix fallback using `scheduler.sigmas` when available
    """
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latents, noise, timesteps)

    if hasattr(scheduler, "scale_noise"):
        try:
            return scheduler.scale_noise(latents, timesteps, noise)
        except TypeError:
            try:
                return scheduler.scale_noise(latents, noise, timesteps)
            except TypeError:
                return scheduler.scale_noise(latents, timesteps)

    if hasattr(scheduler, "sigmas"):
        sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
        t = timesteps.clamp(min=0, max=len(sigmas) - 1)
        sigma = sigmas[t].view(-1, 1, 1, 1)
        return (1.0 - sigma) * latents + sigma * noise

    raise RuntimeError(
        "Scheduler has neither `add_noise` nor `scale_noise`, and no `sigmas` for fallback mixing."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Direct FLUX PGD attack for Anti-DreamBooth-style image perturbation.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir_for_adversarial", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--max_adv_train_steps", type=int, default=6)
    parser.add_argument("--checkpointing_iterations", type=int, default=10)
    parser.add_argument("--pgd_alpha", type=float, default=5e-3)
    parser.add_argument("--pgd_eps", type=float, default=5e-2)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    if args.mixed_precision == "fp32":
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(device)

    if not hasattr(pipe, "vae") or not hasattr(pipe, "transformer") or not hasattr(pipe, "scheduler"):
        raise RuntimeError(
            "This script expects a FLUX-like pipeline with `.vae`, `.transformer`, and `.scheduler`."
        )

    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    data, names = load_images(args.instance_data_dir_for_adversarial, args.resolution, args.center_crop)
    data = data.to(device=device, dtype=torch.float32)
    original = data.clone().detach()
    adv = data.clone().detach()

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_flux_prompt(
        pipe, args.instance_prompt, batch_size=len(data), device=device
    )

    for step in range(args.max_train_steps):
        for _ in range(args.max_adv_train_steps):
            adv.requires_grad_(True)

            latents = pipe.vae.encode(adv.to(dtype=dtype)).latent_dist.sample()
            scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
            latents = latents * scaling_factor

            noise = torch.randn_like(latents)
            if hasattr(pipe.scheduler.config, "num_train_timesteps"):
                max_t = pipe.scheduler.config.num_train_timesteps
            else:
                max_t = 1000
            timesteps = torch.randint(1, max_t, (latents.shape[0],), device=device).long()

            noisy = add_noise_with_fallback(pipe.scheduler, latents, noise, timesteps)

            pred = call_flux_transformer(
                pipe.transformer,
                noisy,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            )

            # Untargeted objective: maximize denoising error
            loss = -F.mse_loss(pred.float(), noise.float(), reduction="mean")
            loss.backward()

            with torch.no_grad():
                adv = adv + args.pgd_alpha * adv.grad.sign()
                delta = torch.clamp(adv - original, min=-args.pgd_eps, max=args.pgd_eps)
                adv = torch.clamp(original + delta, min=-1.0, max=1.0)

            print(f"step={step:04d} inner_loss={loss.detach().item():.6f}")

        if (step + 1) % args.checkpointing_iterations == 0:
            ckpt_dir = os.path.join(args.output_dir, "noise-ckpt", str(step + 1))
            save_images(adv.detach().float().cpu(), names, ckpt_dir, prefix=f"{step+1}_noise")
            print(f"Saved perturbations to: {ckpt_dir}")

    final_dir = os.path.join(args.output_dir, "noise-ckpt", "final")
    save_images(adv.detach().float().cpu(), names, final_dir, prefix="final_noise")
    print(f"Done. Final perturbations saved to: {final_dir}")


if __name__ == "__main__":
    main()
