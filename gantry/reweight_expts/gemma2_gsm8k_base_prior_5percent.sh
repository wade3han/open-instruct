#!/bin/bash
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6
NAME=gemma2_2b_gsm8k_base_prior_v2_lr2e-5

export WANDB_ENTITY='seungjuhan3'
export WANDB_PROJECT='lora_olmo1b_selections'
export WANDB_NAME=$NAME
python open_instruct/gradient/finetune_reweight_gsm8k.py \
  --use_multipack \
  --use_compile \
  --mask_users \
  --model_name_or_path google/gemma-2-2b \
  --use_flash_attn \
  --tokenizer_name google/gemma-2-2b \
  --train_file /net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/round0_data.jsonl \
  --max_seq_length 2048 \
  --preprocessing_num_workers 128 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0. \
  --eval_per_steps 20 \
  --num_train_epochs 1 \
  --output_dir ./debug_results/$NAME \
  --reduce_loss "sum" \
  --lr_scheduler_type "wsd" \
  --cooldown_ratio 0.2 \
  --logging_steps 20 \
  --clip_grad_norm 1.0 \
  --max_train_samples 500 \
  --with_tracking \
  --report_to wandb \
  --validation_dataset_names gsm8k \
  --lora_alpha 128 \
  --max_test_samples 100 \
  --per_device_eval_batch_size 1
