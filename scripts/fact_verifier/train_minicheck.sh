#!/bin/bash
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6

name=minicheck_gemma2_it_seq3072_chat_template
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
  --model_name_or_path google/gemma-2-2b-it \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --tokenizer_name google/gemma-2-2b-it \
  --use_slow_tokenizer \
  --train_file /home/ubuntu/minicheck_dataset.jsonl \
  --max_seq_length 3072 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 5e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0. \
  --num_train_epochs 1 \
  --output_dir $name \
  --report_to wandb \
  --logging_steps 10 \
  --gradient_checkpointing \
  --with_tracking
