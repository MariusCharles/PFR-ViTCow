OUTPUT_DIR="../results/"
DATA_PATH="../pretraining_dataset/train.csv"
VAL_DATA_PATH="../pretraining_dataset/test.csv"

OMP_NUM_THREADS=1 \
python ../../VideoMAE/run_mae_pretraining.py \
  --data_path ${DATA_PATH} \
  --val_data_path ${VAL_DATA_PATH} \
  --val_freq 10 \
  --input_size 224 \
  --mask_type tube \
  --mask_ratio 0.9 \
  --model pretrain_videomae_base_patch16_224 \
  --decoder_depth 4 \
  --drop_path 0.1 \
  --batch_size 64 \
  --use_checkpoint \
  --opt adamw \
  --opt_betas 0.9 0.95 \
  --weight_decay 0.05 \
  --lr 3e-4 \
  --epochs 200 \
  --warmup_epochs 10 \
  --warmup_lr 1e-6 \
  --min_lr 1e-6 \
  --save_ckpt_freq 20 \
  --num_workers 20 \
  --auto_resume \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --wandb \
  --wandb_project videomae-vitcow \
  --wandb_run_name pretrain_70k_videos