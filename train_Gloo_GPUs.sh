torchrun --nproc_per_node=2 --master_port=12355 train_ds.py --backend gloo \
    --log_dir "t5_translation_logs_Gloo_GPUs" \
    --train_epochs 2 \
    --batch_size 256 \
    --eval_batch_size 512 \
    --save_path "checkpoints/t5_Gloo_GPUs" \
    --data_split 0.05
