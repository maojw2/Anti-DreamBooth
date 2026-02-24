diff --git a/attacks/flux_pgd.py b/attacks/flux_pgd.py
index d6ef5b9bde4e847cc7c6f80407b372c6f928b386..b89b247b979768c6e2a53f5f1ec658acc71410b5 100644
--- a/attacks/flux_pgd.py
+++ b/attacks/flux_pgd.py
@@ -45,53 +45,101 @@ def parse_args() -> argparse.Namespace:
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
+    parser.add_argument(
+        "--disable_lora_adapters",
+        action="store_true",
+        help=(
+            "Disable/unload any LoRA adapters after pipeline loading. "
+            "Useful when a mismatched LoRA causes shape errors between FLUX variants."
+        ),
+    )
     return parser.parse_args()
 
 
+def disable_lora_adapters_if_possible(pipe: DiffusionPipeline) -> list[str]:
+    """
+    Best-effort removal of LoRA/adapters to run the base model only.
+
+    This follows the recommended debugging workflow for dimension mismatch issues:
+    first verify the base FLUX model works without any adapter.
+    """
+    actions: list[str] = []
+
+    if hasattr(pipe, "unload_lora_weights"):
+        pipe.unload_lora_weights()
+        actions.append("pipe.unload_lora_weights")
+
+    if hasattr(pipe, "disable_lora"):
+        pipe.disable_lora()
+        actions.append("pipe.disable_lora")
+
+    if hasattr(pipe, "unfuse_lora"):
+        pipe.unfuse_lora()
+        actions.append("pipe.unfuse_lora")
+
+    if hasattr(pipe, "set_adapters"):
+        pipe.set_adapters([])
+        actions.append("pipe.set_adapters([])")
+
+    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "disable_lora"):
+        pipe.transformer.disable_lora()
+        actions.append("pipe.transformer.disable_lora")
+
+    if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "disable_lora"):
+        pipe.text_encoder.disable_lora()
+        actions.append("pipe.text_encoder.disable_lora")
+
+    if hasattr(pipe, "text_encoder_2") and hasattr(pipe.text_encoder_2, "disable_lora"):
+        pipe.text_encoder_2.disable_lora()
+        actions.append("pipe.text_encoder_2.disable_lora")
+
+    return actions
+
+
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
@@ -511,50 +559,57 @@ def main() -> None:
 
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
 
+    if args.disable_lora_adapters:
+        actions = disable_lora_adapters_if_possible(pipe)
+        if actions:
+            print(f"Disabled adapters/LoRA via: {', '.join(actions)}")
+        else:
+            print("--disable_lora_adapters was set, but no adapter control API was found on this pipeline.")
+
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
 
