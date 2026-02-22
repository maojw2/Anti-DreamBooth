import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch FLUX DreamBooth LoRA training through the official diffusers training script."
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default=os.environ.get("FLUX_TRAIN_SCRIPT", ""),
        help=(
            "Path to diffusers FLUX DreamBooth training script (e.g. "
            "diffusers/examples/dreambooth/train_dreambooth_lora_flux.py)."
        ),
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "tensorboard", "wandb"],
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 where supported.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the final accelerate command without launching training.",
    )
    return parser.parse_args()


def resolve_train_script(path_value: str) -> Path:
    candidates = []
    if path_value:
        candidates.append(Path(path_value))

    candidates.extend(
        [
            Path("diffusers/examples/dreambooth/train_dreambooth_lora_flux.py"),
            Path("./examples/dreambooth/train_dreambooth_lora_flux.py"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Cannot find `train_dreambooth_lora_flux.py`. Pass `--train_script` or set FLUX_TRAIN_SCRIPT."
    )


def main():
    args = parse_args()
    train_script = resolve_train_script(args.train_script)

    command = [
        "accelerate",
        "launch",
        str(train_script),
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--instance_data_dir",
        args.instance_data_dir,
        "--output_dir",
        args.output_dir,
        "--instance_prompt",
        args.instance_prompt,
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--lr_scheduler",
        args.lr_scheduler,
        "--lr_warmup_steps",
        str(args.lr_warmup_steps),
        "--max_train_steps",
        str(args.max_train_steps),
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--mixed_precision",
        args.mixed_precision,
        "--seed",
        str(args.seed),
        "--rank",
        str(args.rank),
        "--report_to",
        args.report_to,
    ]

    if args.allow_tf32:
        command.append("--allow_tf32")

    print("[FLUX Train Command]")
    print(" ".join(shlex.quote(x) for x in command))

    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
