#!/bin/bash
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6
NAME=megamixv2_batch4_seq8192_sum_lr5e-5_wsd20_user_mask_a100_tulu2mix-cot

gantry run --beaker-image seungjuh/open-instruct-public-240711 --venv base   --name $NAME   --cluster ai2/general-cirrascale-a100-80g-ib   --workspace ai2/safety   --pip requirements.txt   --workspace ai2/safety   --gpus 4   --priority high   --preemptible   --env-secret WANDB_API_KEY=WANDB_API_KEY   --env-secret HF_TOKEN=HUGGING_FACE_HUB_TOKEN   --env WANDB_PROJECT=llama2-finetuning   --env WANDB_ENTITY=seungjuhan3   --env WANDB_NAME=$NAME   --env-secret OPENAI_API_KEY=openai_api_key   --budget ai2/oe-adapt -- accelerate launch   --mixed_precision bf16   --num_machines 1   --num_processes $NUM_GPUS   --use_deepspeed   --main_process_port 2950   --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf   open_instruct/finetune.py   --use_multipack   --use_compile   --mask_users   --model_name_or_path meta-llama/Llama-2-7b-hf   --use_flash_attn   --tokenizer_name meta-llama/Llama-2-7b-hf   --train_file /net/nfs.cirrascale/mosaic/seungjuh/open-instruct/datasets/megamixv2_dedup_loo_tulu2mix-cot_train.jsonl   --max_seq_length 8192   --preprocessing_num_workers 128   --per_device_train_batch_size $BATCH_SIZE_PER_GPU   --gradient_accumulation_steps $GRADIENT_ACC_STEPS   --learning_rate 5e-5   --warmup_ratio 0.03   --weight_decay 0.   --num_train_epochs 4   --output_dir /results/$NAME   --with_tracking   --report_to wandb   --gradient_checkpointing   --reduce_loss "sum"   --lr_scheduler_type "wsd"   --cooldown_ratio 0.2   --logging_steps 1
