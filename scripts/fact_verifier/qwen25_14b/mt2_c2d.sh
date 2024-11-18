#!/bin/bash
NUM_GPUS=1
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6

name=qwen2_5_14B_mt_cot_3_c2d_alpha128_lr1e-5
accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  --main_process_port 29500 \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --wandb_entity seungjuhan3 \
  --wandb_project fact_verifier_controlled \
  --wandb_name $name \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --tokenizer_name Qwen/Qwen2.5-14B-Instruct \
  --use_slow_tokenizer \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --train_file /home/ubuntu/scalable-factuality/train/train/size_cont_v3_0_1_8000-size_anli_64k_8000.jsonl,/home/ubuntu/scalable-factuality/train/train_cot/size_final_v2_short_reasoning_path_cont_v3_0_1_8000-size_final_v2_short_reasoning_path_anli_8000.jsonl,/home/ubuntu/scalable-factuality/train/train/c2d.jsonl \
  --max_seq_length 1024 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 1e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0. \
  --num_train_epochs 2 \
  --output_dir $name \
  --report_to wandb \
  --eval_file /home/ubuntu/open-instruct-general/fact_verification_dev.jsonl \
  --eval_steps 10000 \
  --gradient_checkpointing \
  --logging_steps 10 \
  --with_tracking
