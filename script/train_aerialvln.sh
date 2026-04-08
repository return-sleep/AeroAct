#!/usr/bin/env bash
# Please modify the data paths in llava/data/datasets_mixture.py and Please modify the data paths in script/train_aerialvln.sh
data_mixture=$1 # airvln_merge_triple_updated
num_video_frames=8 # sample num_video_frames from hisotry_queue_len
temporal_aggregation_frames=8
idx_type=merge # default means continuous timestep, merge means keyframes when merged action happens
hisotry_queue_len=0 # max hisotry_queue_len 0 means all history,  8 means last 8
label_balance_type=v1
num_train_epochs=1
train_ratio=1.0
seed=10
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT="./checkpoints/exp1"
GPUS_PER_NODE=${GPUS_PER_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}
echo "Using $GPUS_PER_NODE GPUs"
torchrun --standalone --nproc-per-node="${GPUS_PER_NODE}" \
  llava/train/train_mem.py \
  --longvila_sampler True \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path Efficient-Large-Model/NVILA-Lite-8B  \
  --seed $seed \
  --data_mixture $data_mixture \
  --mm_vision_select_feature cls_patch \
  --num_video_frames $num_video_frames \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio dynamic \
  --bf16 True \
  --output_dir "$OUTPUT" \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --do_eval False \
  --save_strategy steps \
  --save_steps 500 \
  --fps 0.0 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --report_to tensorboard \
  --train_ratio $train_ratio \
  --idx_type $idx_type \
  --hisotry_queue_len $hisotry_queue_len \
  --temporal_aggregation_frames $temporal_aggregation_frames\
  --label_balance \
  --label_balance_type $label_balance_type \
  --tune_mm_projector True \
  --tune_language_model True \
  --tune_vision_tower True \
  