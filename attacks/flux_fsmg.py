import argparse

from flux_pgd import main as run_flux_pgd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anti-DreamBooth FSMG-style attack for FLUX models (direct denoiser objective)."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Directory containing images to perturb.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")

    # Keep option names consistent with FSMG scripts.
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--max_adv_train_steps", type=int, default=6)
    parser.add_argument("--checkpointing_steps", type=int, default=20)

    parser.add_argument("--attack_batch_size", type=int, default=1)
    parser.add_argument("--pgd_alpha", type=float, default=5e-3)
    parser.add_argument("--pgd_eps", type=float, default=5e-2)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def remap_to_flux_pgd_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--instance_data_dir_for_adversarial",
        args.instance_data_dir,
        "--output_dir",
        args.output_dir,
        "--instance_prompt",
        args.instance_prompt,
        "--resolution",
        str(args.resolution),
        "--max_train_steps",
        str(args.max_train_steps),
        "--max_adv_train_steps",
        str(args.max_adv_train_steps),
        "--checkpointing_iterations",
        str(args.checkpointing_steps),
        "--attack_batch_size",
        str(args.attack_batch_size),
        "--pgd_alpha",
        str(args.pgd_alpha),
        "--pgd_eps",
        str(args.pgd_eps),
        "--mixed_precision",
        args.mixed_precision,
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
    ]
    if args.center_crop:
        argv.append("--center_crop")
    if args.local_files_only:
        argv.append("--local_files_only")
    return argv


def main() -> None:
    args = parse_args()
    argv = remap_to_flux_pgd_argv(args)

    # Reuse robust FLUX compatibility logic in attacks/flux_pgd.py.
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["flux_fsmg.py", *argv]
        run_flux_pgd()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
