import argparse
import os

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline


parser = argparse.ArgumentParser(description="Inference")
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./test-infer/",
    help="The output directory where predictions are saved",
)
parser.add_argument(
    "--model_family",
    type=str,
    choices=["sd", "flux"],
    default="sd",
    help="Model family to use for inference.",
)
parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=100,
    help="Number of denoising steps.",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="Classifier-free guidance scale.",
)

args = parser.parse_args()


def load_pipeline(model_path: str, model_family: str):
    load_kwargs = {
        "torch_dtype": torch.float16,
        "local_files_only": True,
    }

    if model_family == "sd":
        return StableDiffusionPipeline.from_pretrained(
            model_path,
            safety_checker=None,
            **load_kwargs,
        ).to("cuda")

    if model_family == "flux":
        return DiffusionPipeline.from_pretrained(
            model_path,
            **load_kwargs,
        ).to("cuda")

    raise ValueError(f"Unsupported model_family: {model_family}")


if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)

    # define prompts
    prompts = [
        "a photo of a sks person",
        "a dslr portrait of sks person",
        "a close-up photo of sks person riding a bike",
        "a photo of sks person in front of eiffel tower",
        "a selfie photo of sks person on top of mount fuji",
    ]

    # create & load model
    pipe = load_pipeline(args.model_path, args.model_family)

    for prompt in prompts:
        print(">>>>>>", prompt)
        norm_prompt = prompt.lower().replace(",", "").replace(" ", "_")
        out_path = f"{args.output_dir}/{norm_prompt}"
        os.makedirs(out_path, exist_ok=True)
        for i in range(2):
            images = pipe(
                [prompt] * 8,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images
            for idx, image in enumerate(images):
                image.save(f"{out_path}/{i}_{idx}.png")
    del pipe
    torch.cuda.empty_cache()
