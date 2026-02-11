OUTPUT_DIR="/teamspace/studios/this_studio/PFR-ViTCow/results/pretrain_template"
DATA_PATH="/teamspace/studios/this_studio/PFR-ViTCow/pretraining_dataset/train.csv"
NUM_GPUS=8
NUM_NODES=8
NODE_RANK=0
MASTER_ADDR=$ip_node_0
MASTER_PORT=12320

OMP_NUM_THREADS=1 \
python -m torch.distributed.launch \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${NUM_NODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ../VideoMAE/run_mae_pretraining.py \
  --data_path ${DATA_PATH} \
  --use_decord \
  --num_frames 16 \
  --sampling_rate 2 \
  --input_size 224 \
  --short_side_size 224 \
  --repeated_aug \
  --mask_type tube \
  --mask_ratio 0.9 \
  --model pretrain_videomae_base_patch16_224 \
  --decoder_depth 4 \
  --decoder_embed_dim 384 \
  --drop_path 0.0 \
  --batch_size 32 \
  --opt adamw \
  --opt_betas 0.9 0.95 \
  --weight_decay 0.05 \
  --lr 1.5e-4 \
  --epochs 801 \
  --warmup_epochs 40 \
  --warmup_lr 1e-6 \
  --min_lr 1e-6 \
  --use_fp16 \
  --loss_scale 65536 \
  --save_ckpt_freq 20 \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR}