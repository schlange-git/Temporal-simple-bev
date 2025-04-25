#!/bin/bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

#DATA_DIR="/mnt/data/sets/nuscenes"
DATA_DIR="/mnt/home/data/sets/nuscenes"

# there should be ${DATA_DIR}/full_v1.0/
# and also ${DATA_DIR}/mini

EXP_NAME="rgb00" # default settings
EXP_NAME="rgb01" # removed some code duplication
EXP_NAME="rgb02" # cleaned up dataset file
EXP_NAME="rgb03" # updated log dir

python train_nuscenes.py \
       --exp_name=${EXP_NAME} \
       --max_iters=25000 \
       --log_freq=1000 \
       --shuffle=True \
       --val_freq=50\
       --save_freq=1000\
       --dset='trainval' \
       --batch_size=1 \
       --grad_acc=5\
       --data_dir=$DATA_DIR \
       --log_dir='logs_nuscenes' \
       --ckpt_dir='checkpoints' \
       --seqlen=2 \
       --res_scale=1 \
       --rand_flip=False \
       --rand_crop_and_resize=True \
       --ncams=6 \
       --encoder_type='res101' \
       --do_rgbcompress=True \
       --device_ids=[0] 

