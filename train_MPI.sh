torchrun --nproc_per_node=2 --master_port=12355 train_ds.py --backend mpi \
    --log_dir "t5_translation_logs_mpi" \
    --train_epochs 2 \
    --batch_size 256 \
    --eval_batch_size 512 \
    --save_path "/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_4_xzp/checkpoints/t5_MPI" \
    --data_split 0.05