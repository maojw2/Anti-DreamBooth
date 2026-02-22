export FLUX_TRAIN_SCRIPT="third_party/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py"

export MODEL_PATH="stable-diffusion/flux-dev"
export INSTANCE_DIR="data/n000050/"
export FLUX_OUTPUT_DIR="flux-dreambooth-outputs/n000050/"

python train_flux_dreambooth_lora.py \
  --train_script="$FLUX_TRAIN_SCRIPT" \
  --download_missing_script \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$FLUX_OUTPUT_DIR" \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --mixed_precision=bf16 \
  --rank=16 \
  --report_to="none"
