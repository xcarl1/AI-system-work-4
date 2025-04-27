torchrun --nproc_per_node=2 --master_port=12355 train_ds.py --backend nccl \
    --log_dir "t5_translation_logs_NCCL" \
    --train_epochs 2 \
    --batch_size 256 \
    --eval_batch_size 512 \
    --save_path "checkpoints/t5_NCCL" \
    --data_split 0.05 \

