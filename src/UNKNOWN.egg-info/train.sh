deepspeed --include localhost:2,3,4,5 --master_port 29501 train.py \
  --deepspeed_config ds_config.json \
  --output_dir outputs/VAE_3_4_watermark \
  --dataset /data2/workspace/bydeng/DATASETS/ALIBABA_DATA/Alibaba-poster-text_watermark/train \
  --batch_size 3 \
  --num_epochs 2 \
  --image_size 512 \
  --lr 5e-4 \
  --save_steps 3000 \
  --load_resume '/data2/workspace/bydeng/Projects/VAE/outputs/VAE_3_4_watermark/weights/final/'