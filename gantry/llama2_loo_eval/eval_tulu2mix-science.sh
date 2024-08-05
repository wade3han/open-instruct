#!/bin/bash
NUM_GPUS=4
echo "Evaluating llama model using $NUM_GPUS GPUs"
# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# sweep learning rate from 2e-5 to 1e-6
NAME=loo-v2_megamixv2_batch4_seq8192_sum_lr5e-5_wsd20_user_mask_a100_tulu2mix-science

gantry run --beaker-image seungjuh/open-instruct-public-240711 --venv base   --name $NAME-evaluation   --cluster ai2/pluto-cirrascale   --workspace ai2/safety   --dataset $NAME:/model   --pip requirements.txt   --gpus $NUM_GPUS   --priority high   --preemptible   --env-secret WANDB_API_KEY=WANDB_API_KEY   --env-secret HF_TOKEN=HUGGING_FACE_HUB_TOKEN   --env WANDB_PROJECT=llama2-evaluation   --env WANDB_ENTITY=seungjuhan3   --env WANDB_NAME=$NAME-evaluation   --env-secret OPENAI_API_KEY=openai_api_key   --budget ai2/oe-adapt -- accelerate launch   --mixed_precision bf16   --num_machines 1   --num_processes $NUM_GPUS   --use_deepspeed   --main_process_port 2950   --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf   open_instruct/evaluate.py   --model_name_or_path /model/$NAME   --use_flash_attn   --tokenizer_name /model/$NAME   --max_seq_length 8192   --train_file 'dummy'   --preprocessing_num_workers 128   --output_dir /results/$NAME   --with_tracking   --report_to wandb
