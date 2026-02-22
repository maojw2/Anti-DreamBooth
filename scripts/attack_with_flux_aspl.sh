export EXPERIMENT_NAME="ASPL_FLUX"
export ATTACK_MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
export FLUX_MODEL_PATH="stable-diffusion/flux-dev"

export CLEAN_TRAIN_DIR="data/n000050/set_A"
export CLEAN_ADV_DIR="data/n000050/set_B"
export CLASS_DIR="data/class-person"

export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"
mkdir -p "$OUTPUT_DIR"
cp -r "$CLEAN_TRAIN_DIR" "$OUTPUT_DIR/image_clean"
cp -r "$CLEAN_ADV_DIR" "$OUTPUT_DIR/image_before_addding_noise"

# 1) Generate Anti-DreamBooth perturbations.
accelerate launch attacks/aspl.py \
  --pretrained_model_name_or_path="$ATTACK_MODEL_PATH" \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train="$CLEAN_TRAIN_DIR" \
  --instance_data_dir_for_adversarial="$CLEAN_ADV_DIR" \
  --instance_prompt="a photo of sks person" \
  --class_data_dir="$CLASS_DIR" \
  --num_class_images=200 \
  --class_prompt="a photo of person" \
  --output_dir="$OUTPUT_DIR" \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2

# 2) Train FLUX DreamBooth LoRA on the perturbed images.
export PERTURBED_INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/50"
export FLUX_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_FLUX_DREAMBOOTH"

python train_flux_dreambooth_lora.py \
  --train_script third_party/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py \
  --download_missing_script \
  --pretrained_model_name_or_path "$FLUX_MODEL_PATH" \
  --instance_data_dir "$PERTURBED_INSTANCE_DIR" \
  --output_dir "$FLUX_OUTPUT_DIR" \
  --instance_prompt "a photo of sks person" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 1000 \
  --checkpointing_steps 500 \
  --mixed_precision bf16 \
  --rank 16
