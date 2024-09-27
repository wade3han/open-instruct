#!/bin/bash
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6

name=internlm_v15_1_minicheck
accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --wandb_entity seungjuhan3 \
  --wandb_project fact_verifier \
  --wandb_name $name \
  --model_name_or_path internlm/internlm2_5-7b-chat \
  --tokenizer_name internlm/internlm2_5-7b-chat \
  --trust_remote_code \
  --use_slow_tokenizer \
  --train_file /home/ubuntu/v15_1_minicheck.jsonl \
  --max_seq_length 2048 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0. \
  --num_train_epochs 2 \
  --output_dir $name \
  --report_to wandb \
  --eval_file /home/ubuntu/open-instruct-general/eval.jsonl \
  --eval_steps 100 \
  --gradient_checkpointing \
  --logging_steps 25 \
  --with_tracking
