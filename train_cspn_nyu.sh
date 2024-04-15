#!/bin/bash
# use this only for local usage
# export CUDA_VISIBLE_DEVICES=0

data_set="nyudepth"
n_sample=100
train_list="datalist/png_train_test.csv"
eval_list="datalist/png_train_test.csv"
model="cspn_unet"

batch_size_train=1
num_epoch_train=1
batch_size_eval=1
# model_name=nyu_pretrain_cspn_1_net_cp500_bs8_adlr_ep40_8norm
model_name=test
save_dir="output/${model_name}"
best_model_dir="output/${model_name}"

python train.py \
--data_set $data_set \
--n_sample $n_sample \
--train_list $train_list \
--eval_list $eval_list \
--model $model \
--batch_size_train $batch_size_train \
--batch_size_eval $batch_size_eval \
--num_epoch $num_epoch_train \
--save_dir $save_dir \
--best_model_dir $best_model_dir \
-n \
-p \
