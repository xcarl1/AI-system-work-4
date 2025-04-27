#!/bin/bash

NUM_PROCESSES=64  # 启动的进程数，可以根据你CPU核数设置

python -m torch.distributed.run \
    --nproc_per_node=$NUM_PROCESSES \
    --master_port=12355 \
    train_ds.py \
    --backend gloo \
    --batch_size 128 \
    --eval_batch_size 512 \
    --train_epochs 2 \
    --lr 1e-5 \
    --log_dir t5_translation_logs_Gloo_CPUs \
    --save_path "/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_4_xzp/checkpoints/t5_Gloo_CPUs" \
