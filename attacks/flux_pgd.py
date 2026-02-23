import argparse
import inspect
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from PIL import Image
from torchvision import transforms


def resolve_flux_pipeline_class():
    """
    Resolve FluxPipeline class across diffusers versions.

    - Newer versions may expose `FluxPipeline` at top-level `diffusers`.
    - Some versions only provide it in `diffusers.pipelines.flux.pipeline_flux`.
    """
    try:
        from diffusers import FluxPipeline  # type: ignore

        return FluxPipeline
    except Exception:
        pass

    try:
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # type: ignore

        return FluxPipeline
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct FLUX PGD attack for Anti-DreamBooth-style image perturbation."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir_for_adversarial", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--max_adv_train_steps", type=int, default=6)
    parser.add_argument(
        "--attack_batch_size",
        type=int,
        default=1,
        help="Number of images optimized per inner PGD step to reduce VRAM usage.",
    )
    parser.add_argument("--checkpointing_iterations", type=int, default=10)
    parser.add_argument("--pgd_alpha", type=float, default=5e-3)
    parser.add_argument("--pgd_eps", type=float, default=5e-2)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance value used when the FLUX transformer forward expects `guidance`.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="If set, do not try downloading from Hub when loading the FLUX pipeline.",
    )
    return parser.parse_args()


@torch.no_grad()
def load_images(data_dir: str, size: int, center_crop: bool) -> tuple[torch.Tensor, list[str]]:
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {data_dir}")

    image_paths = sorted([p for p in data_root.iterdir() if p.is_file()])
    if not image_paths:
        raise ValueError(f"No images found in: {data_dir}")

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


def save_images(images: torch.Tensor, names: list[str], out_dir: str, prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(images, names):
        arr = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(os.path.join(out_dir, f"{prefix}_{name}"))


def _call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]):
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return fn(**filtered)


def prepare_flux_hidden_and_img_ids(pipe: DiffusionPipeline, latents: torch.Tensor):
    """
    Prepare transformer hidden states / image ids across FLUX pipeline variants.
    """
    hidden_states = latents
    img_ids = None

    if hasattr(pipe, "_pack_latents"):
        hidden_states = _call_with_supported_kwargs(
            pipe._pack_latents,
            {
                "latents": latents,
                "batch_size": latents.shape[0],
                "num_channels_latents": latents.shape[1],
                "height": latents.shape[-2],
                "width": latents.shape[-1],
            },
        )

    if hasattr(pipe, "_prepare_latent_image_ids"):
        h = latents.shape[-2]
        w = latents.shape[-1]
        img_ids = _call_with_supported_kwargs(
            pipe._prepare_latent_image_ids,
            {
                "batch_size": latents.shape[0],
                "height": h,
                "width": w,
                "device": latents.device,
                "dtype": latents.dtype,
            },
        )

    return hidden_states, img_ids


def encode_flux_prompt(pipe: DiffusionPipeline, prompt: str, batch_size: int, device: torch.device):
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("Loaded pipeline does not support `encode_prompt`, cannot run FLUX attack.")

    encoded = _call_with_supported_kwargs(
        pipe.encode_prompt,
        {
            "prompt": [prompt] * batch_size,
            "prompt_2": [prompt] * batch_size,
            "device": device,
            "num_images_per_prompt": 1,
            "max_sequence_length": 512,
        },
    )

    # Common FLUX return: (prompt_embeds, pooled_prompt_embeds, text_ids)
    # but order can differ across versions.
    if isinstance(encoded, tuple):
        if len(encoded) >= 3:
            prompt_embeds = None
            pooled_prompt_embeds = None
            text_ids = None

            for item in encoded:
                if not torch.is_tensor(item):
                    continue

                # Prompt embeds are usually 3D floating tensors [B, T, D].
                if item.ndim >= 3 and torch.is_floating_point(item) and prompt_embeds is None:
                    prompt_embeds = item
                    continue

                # Pooled projection is usually 2D floating tensor [B, D].
                if item.ndim == 2 and torch.is_floating_point(item) and pooled_prompt_embeds is None:
                    pooled_prompt_embeds = item
                    continue

                # text ids are typically integer tensors.
                if not torch.is_floating_point(item) and text_ids is None:
                    text_ids = item

            # Fallback to positional assumptions if still missing.
            if prompt_embeds is None:
                prompt_embeds = encoded[0]
            if pooled_prompt_embeds is None:
                pooled_prompt_embeds = encoded[1]
            if text_ids is None and len(encoded) > 2:
                text_ids = encoded[2]

            if pooled_prompt_embeds is None and torch.is_tensor(prompt_embeds) and prompt_embeds.ndim >= 3:
                # Some variants don't return pooled projection explicitly.
                # Use mean pooling as a safe fallback to satisfy transformer conditioning input.
                pooled_prompt_embeds = prompt_embeds.mean(dim=1)

            return prompt_embeds, pooled_prompt_embeds, text_ids
        if len(encoded) == 2:
            return encoded[0], encoded[1], None
        if len(encoded) == 1:
            return encoded[0], None, None

    if isinstance(encoded, dict):
        prompt_embeds = encoded.get("prompt_embeds")
        if prompt_embeds is None:
            prompt_embeds = encoded.get("encoder_hidden_states")

        pooled_prompt_embeds = encoded.get("pooled_prompt_embeds")
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = encoded.get("pooled_projections")
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = encoded.get("pooled_projection")

        text_ids = encoded.get("text_ids")
        if text_ids is None:
            text_ids = encoded.get("txt_ids")

        if pooled_prompt_embeds is None and torch.is_tensor(prompt_embeds) and prompt_embeds.ndim >= 3:
            pooled_prompt_embeds = prompt_embeds.mean(dim=1)

        return (prompt_embeds, pooled_prompt_embeds, text_ids)

    raise RuntimeError("Unsupported return format from `encode_prompt`.")


def call_flux_transformer(
    transformer,
    noisy_latents,
    timesteps,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    guidance_scale: float,
    img_ids_override=None,
):
    if pooled_prompt_embeds is None and torch.is_tensor(prompt_embeds) and prompt_embeds.ndim >= 3:
        pooled_prompt_embeds = prompt_embeds.mean(dim=1)

    sig = inspect.signature(transformer.forward)

    # FLUX variants may require txt_ids/img_ids for RoPE indexing.
    # Build safe fallbacks if missing.
    if text_ids is None and ("txt_ids" in sig.parameters or "text_ids" in sig.parameters):
        text_ids = torch.zeros((timesteps.shape[0], 1, 3), device=noisy_latents.device, dtype=torch.long)

    img_ids = img_ids_override
    if img_ids is None and "img_ids" in sig.parameters:
        if noisy_latents.ndim >= 3:
            img_tokens = noisy_latents.shape[1] if noisy_latents.ndim == 3 else noisy_latents.shape[-2] * noisy_latents.shape[-1]
        else:
            img_tokens = 1

        id_last_dim = 3
        id_dtype = torch.long
        if torch.is_tensor(text_ids):
            id_dtype = text_ids.dtype
            if text_ids.ndim >= 3:
                id_last_dim = text_ids.shape[-1]

        img_ids = torch.zeros((timesteps.shape[0], img_tokens, id_last_dim), device=noisy_latents.device, dtype=id_dtype)

    guidance = None
    if "guidance" in sig.parameters:
        # Some FLUX variants require a guidance tensor in forward;
        # if omitted, internal time-text embedding can fail with missing pooled_projection.
        guidance = torch.full(
            (timesteps.shape[0],),
            float(guidance_scale),
            dtype=noisy_latents.dtype,
            device=noisy_latents.device,
        )

    out = _call_with_supported_kwargs(
        transformer.forward,
        {
            "hidden_states": noisy_latents,
            "sample": noisy_latents,
            "timestep": timesteps,
            "timesteps": timesteps,
            "encoder_hidden_states": prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "pooled_projection": pooled_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "txt_ids": text_ids,
            "text_ids": text_ids,
            "img_ids": img_ids,
            "guidance": guidance,
        },
    )

    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    return out


def add_noise_with_fallback(scheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    # 1) DDPM-style schedulers
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latents, noise, timesteps)

    # 2) Flow-matching schedulers
    if hasattr(scheduler, "scale_noise"):
        try:
            return _call_with_supported_kwargs(
                scheduler.scale_noise,
                {
                    "sample": latents,
                    "samples": latents,
                    "latents": latents,
                    "noise": noise,
                    "timesteps": timesteps,
                    "timestep": timesteps,
                },
            )
        except TypeError:
            pass

    # 3) Sigma-based manual fallback
    if hasattr(scheduler, "sigmas"):
        sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
        t = timesteps.clamp(min=0, max=len(sigmas) - 1)
        sigma = sigmas[t].view(-1, 1, 1, 1)
        return (1.0 - sigma) * latents + sigma * noise

    raise RuntimeError(
        "Scheduler has neither `add_noise` nor `scale_noise`, and no `sigmas` for fallback mixing."
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float16
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp32":
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flux_pipeline_cls = resolve_flux_pipeline_class()
    if flux_pipeline_cls is None:
        raise RuntimeError(
            "Your installed diffusers version does not provide FLUX support (`FluxPipeline`). "
            "Please upgrade diffusers to a version that includes FLUX pipelines, e.g.\n"
            "  pip install -U 'diffusers>=0.30.0' transformers accelerate\n"
            "Then rerun this script."
        )

    pipe = flux_pipeline_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.to(device)

    if not hasattr(pipe, "vae") or not hasattr(pipe, "transformer") or not hasattr(pipe, "scheduler"):
        raise RuntimeError("This script expects a FLUX-like pipeline with `.vae`, `.transformer`, and `.scheduler`.")

    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    data, names = load_images(args.instance_data_dir_for_adversarial, args.resolution, args.center_crop)
    data = data.to(device=device, dtype=torch.float32)

    original = data.detach().clone()
    adv = data.detach().clone()

    prompt_embeds, pooled_prompt_embeds, text_ids = encode_flux_prompt(
        pipe, args.instance_prompt, batch_size=len(data), device=device
    )

    for step in range(args.max_train_steps):
        for _ in range(args.max_adv_train_steps):
            if args.attack_batch_size <= 0:
                raise ValueError("--attack_batch_size must be >= 1")

            batch_size = min(args.attack_batch_size, adv.shape[0])
            batch_indices = torch.randperm(adv.shape[0], device=device)[:batch_size]
            adv_batch = adv[batch_indices].detach().clone().requires_grad_(True)

            latents = pipe.vae.encode(adv_batch.to(dtype=dtype)).latent_dist.sample()
            latents = latents * getattr(pipe.vae.config, "scaling_factor", 1.0)

            noise = torch.randn_like(latents)
            if hasattr(pipe.scheduler, "config") and hasattr(pipe.scheduler.config, "num_train_timesteps"):
                max_t = pipe.scheduler.config.num_train_timesteps
            else:
                max_t = 1000
            timesteps = torch.randint(1, max_t, (latents.shape[0],), device=device).long()

            noisy_latents = add_noise_with_fallback(pipe.scheduler, latents, noise, timesteps)
            transformer_hidden, transformer_img_ids = prepare_flux_hidden_and_img_ids(pipe, noisy_latents)

            pred = call_flux_transformer(
                pipe.transformer,
                transformer_hidden,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
                args.guidance_scale,
                img_ids_override=transformer_img_ids,
            )

            loss = -F.mse_loss(pred.float(), noise.float(), reduction="mean")
            loss.backward()

            with torch.no_grad():
                adv_batch = adv_batch + args.pgd_alpha * adv_batch.grad.sign()
                delta = torch.clamp(adv_batch - original[batch_indices], min=-args.pgd_eps, max=args.pgd_eps)
                adv_batch = torch.clamp(original[batch_indices] + delta, min=-1.0, max=1.0)
                adv[batch_indices] = adv_batch

            print(f"step={step:04d} inner_loss={loss.detach().item():.6f}")

            del adv_batch, latents, noise, noisy_latents, transformer_hidden, transformer_img_ids, pred, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (step + 1) % args.checkpointing_iterations == 0:
            out_dir = os.path.join(args.output_dir, "noise-ckpt", str(step + 1))
            save_images(adv.detach().float().cpu(), names, out_dir, prefix=f"{step+1}_noise")
            print(f"Saved perturbations to: {out_dir}")

    final_dir = os.path.join(args.output_dir, "noise-ckpt", "final")
    save_images(adv.detach().float().cpu(), names, final_dir, prefix="final_noise")
    print(f"Done. Final perturbations saved to: {final_dir}")


if __name__ == "__main__":
    main()

