export MODEL_PATH="./stable-diffusion/flux-dev"
export OUTPUT_DIR="./test-infer-flux"

python infer.py \
  --model_family flux \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_inference_steps 50 \
  --guidance_scale 3.5
