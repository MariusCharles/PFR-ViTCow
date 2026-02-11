OUTPUT_DIR="/teamspace/studios/this_studio/PFR-ViTCow/results/testing_1GPU_1EPOCH"
DATA_PATH="/teamspace/studios/this_studio/PFR-ViTCow/pretraining_dataset/train.csv"

OMP_NUM_THREADS=1 \
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  ../VideoMAE/run_mae_pretraining.py \
  --data_path ${DATA_PATH} \
  --mask_type tube \
  --mask_ratio 0.9 \
  --model pretrain_videomae_base_patch16_224 \
  --decoder_depth 4 \
  --batch_size 2 \
  --num_frames 16 \
  --sampling_rate 2 \
  --epochs 1 \
  --warmup_epochs 0 \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR}
