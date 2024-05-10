@echo off
rem use this only for local usage
rem set CUDA_VISIBLE_DEVICES=0

set data_set=nyudepth
set n_sample=100
set train_list=datalist\FallingThings_train.csv
set eval_list=datalist\FallingThings_val.csv
set model=cspn_unet

set batch_size_train=1
set num_epoch_train=100
set batch_size_eval=1
set learning_rate=0.001
set model_name=adam0510_step24_FT200_normalized_resnet18_edited2048_traindepth
set save_dir=output\%model_name%
set best_model_dir=output\%model_name%

set cspn_step=24

python train.py ^
--data_set %data_set% ^
--n_sample %n_sample% ^
--train_list %train_list% ^
--eval_list %eval_list% ^
--model %model% ^
--batch_size_train %batch_size_train% ^
--batch_size_eval %batch_size_eval% ^
--num_epoch %num_epoch_train% ^
--save_dir %save_dir% ^
--best_model_dir %best_model_dir% ^
--cspn_step %cspn_step% ^
--lr %learning_rate% ^
-n -r



