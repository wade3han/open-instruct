#!/bin/bash
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=4
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6

name=llama32_1B_cont_v3_0_1_8k_anli_8k_c2d_2eps_2048_seed1
accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  --main_process_port 29503 \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --seed 1 \
  --wandb_entity seungjuhan3 \
  --wandb_project fact_verifier_controlled \
  --wandb_name $name \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name meta-llama/Llama-3.2-1B-Instruct \
  --use_slow_tokenizer \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --train_file /home/ubuntu/scalable-factuality/train/train/size_cont_v3_0_1_8000-size_anli_64k_8000-c2d.jsonl \
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
  --eval_file /home/ubuntu/open-instruct-general/fact_verification_dev.jsonl \
  --eval_steps 10000 \
  --logging_steps 10 \
  --with_tracking

name=llama32_1B_cont_v3_0_1_8k_anli_8k_c2d_2eps_2048_seed2
accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  --main_process_port 29503 \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --seed 2 \
  --wandb_entity seungjuhan3 \
  --wandb_project fact_verifier_controlled \
  --wandb_name $name \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name meta-llama/Llama-3.2-1B-Instruct \
  --use_slow_tokenizer \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --train_file /home/ubuntu/scalable-factuality/train/train/size_cont_v3_0_1_8000-size_anli_64k_8000-c2d.jsonl \
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
  --eval_file /home/ubuntu/open-instruct-general/fact_verification_dev.jsonl \
  --eval_steps 10000 \
  --logging_steps 10 \
  --with_tracking

name=llama32_1B_cont_v3_0_1_8k_anli_8k_c2d_2eps_2048_seed42
accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  --main_process_port 29503 \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  open_instruct/finetune.py \
  --seed 42 \
  --wandb_entity seungjuhan3 \
  --wandb_project fact_verifier_controlled \
  --wandb_name $name \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer_name meta-llama/Llama-3.2-1B-Instruct \
  --use_slow_tokenizer \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --train_file /home/ubuntu/scalable-factuality/train/train/size_cont_v3_0_1_8000-size_anli_64k_8000-c2d.jsonl \
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
  --eval_file /home/ubuntu/open-instruct-general/fact_verification_dev.jsonl \
  --eval_steps 10000 \
  --logging_steps 10 \
  --with_tracking
