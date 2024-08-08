#!/bin/bash
NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6
NAME=mm_cooldown_deita_complexity_batch4_seq8192_sum_lr5e-5_user_mask

gantry run --beaker-image seungjuh/open-instruct-public-240711 --venv base \
  --name $NAME \
  --cluster ai2/pluto-cirrascale \
  --workspace ai2/safety \
  --pip requirements.txt \
  --workspace ai2/safety \
  --gpus 8 \
  --dataset '01J41RDY5H3HJD1P2Y40W86N87:/model' \
  --priority high \
  --preemptible \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret HF_TOKEN=HUGGING_FACE_HUB_TOKEN \
  --env WANDB_PROJECT=llama2-finetuning \
  --env WANDB_ENTITY=seungjuhan3 \
  --env WANDB_NAME=$NAME \
  --env-secret OPENAI_API_KEY=openai_api_key \
  --budget ai2/oe-adapt -- accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes $NUM_GPUS \
  --use_deepspeed \
  --main_process_port 2950 \
  --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf \
  open_instruct/finetune_accelerate.py \
  --use_multipack \
  --use_compile \
  --mask_users \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --use_flash_attn \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --train_file /net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/megamix_minhash-3gram-0_5_train/deita_processed_top600000_complexity.jsonl \
  --max_seq_length 8192 \
  --preprocessing_num_workers 128 \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 5e-5 \
  --warmup_ratio 0. \
  --weight_decay 0. \
  --num_train_epochs 1 \
  --load_from_checkpoint /model/dacc98e_megamixdedupv1_batch4_seq8192_sum_lr5e-5_wsd0_user_mask \
  --output_dir /results/$NAME \
  --with_tracking \
  --report_to wandb \
  --gradient_checkpointing \
  --reduce_loss "sum" \
  --lr_scheduler_type "wsd" \
  --cooldown_ratio 1.0 \
  --logging_steps 1
