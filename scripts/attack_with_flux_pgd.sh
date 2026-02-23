export EXPERIMENT_NAME="FLUX_PGD"
export FLUX_MODEL_PATH="stable-diffusion/flux-dev"

export CLEAN_ADV_DIR="data/n000050/set_B"

export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"
mkdir -p "$OUTPUT_DIR"
cp -r "$CLEAN_ADV_DIR" "$OUTPUT_DIR/image_before_adding_noise"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --dynamo_backend no \
  attacks/flux_pgd.py \
  --pretrained_model_name_or_path="$FLUX_MODEL_PATH" \
  --instance_data_dir_for_adversarial="$CLEAN_ADV_DIR" \
  --instance_prompt="a photo of sks person" \
  --output_dir="$OUTPUT_DIR" \
  --center_crop \
  --resolution=512 \
  --max_train_steps=50 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2 \
  --mixed_precision=bf16
