#!/usr/bin/bash

OUTPUT_DIR="/workspace/Oscar/logs"
DATASET_DIR="/workspace/Oscar/data"

mkdir -p ${OUTPUT_DIR}

python oscar/run_oscarplus_pretrain.py \
    --not_use_data_parallel \
    --use_b 1 \
    --max_grad_norm 10.0 --gradient_accumulation_steps 8 \
    --use_img_layernorm 1 \
    --output_dir ${OUTPUT_DIR} \
    --bert_model bert --model_name_or_path bert-base-uncased \
    --do_lower_case --learning_rate 5e-05 \
    --warmup_steps 0 --do_train --max_seq_length 35 --on_memory \
    --max_img_seq_length 50 --img_feature_dim 2054 \
    --drop_out 0.1 --train_batch_size 1024 \
    --ckpt_period 10000 --max_iters 2000000 --log_period 100 \
    --use_fakedata \
    --num_workers 1 \
    --textb_sample_mode 1 --texta_false_prob 0.25 \
    --from_scratch