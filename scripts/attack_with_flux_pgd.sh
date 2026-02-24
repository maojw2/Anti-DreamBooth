diff --git a/scripts/attack_with_flux_pgd.sh b/scripts/attack_with_flux_pgd.sh
index 2095f95a7e1398e796a6fbb75ffed45676067b99..421ed0cfbe063774076cdd405d67a6e1e37d8388 100644
--- a/scripts/attack_with_flux_pgd.sh
+++ b/scripts/attack_with_flux_pgd.sh
@@ -1,22 +1,23 @@
 export EXPERIMENT_NAME="FLUX_PGD"
 export FLUX_MODEL_PATH="stable-diffusion/flux-dev"
 
 export CLEAN_ADV_DIR="data/n000050/set_B"
 
 export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"
 mkdir -p "$OUTPUT_DIR"
 cp -r "$CLEAN_ADV_DIR" "$OUTPUT_DIR/image_before_adding_noise"
 
 accelerate launch attacks/flux_pgd.py \
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
-  --mixed_precision=bf16
+  --mixed_precision=bf16 \
+  --disable_lora_adapters
