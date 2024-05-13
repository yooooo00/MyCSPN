@echo off

set data_set=nyudepth
set n_sample=500
@REM set eval_list=datalist\FallingThings_train_less.csv
set eval_list=datalist\FallingThings_val.csv
set model=cspn_unet
set batch_size_eval=1

rem for positive affinity
rem set best_model_dir=output\nyu_pretrain_cspn_1_net_cp500_bs8_adlr_ep40
rem set cspn_norm_type=8sum_abs  


rem for non-positive affinity
set best_model_dir=output\adam0513_train_refine
set cspn_norm_type=8sum
set cspn_step=24
set resume_model_name=best_train_26.22.pth

python eval.py ^
--data_set %data_set% ^
--n_sample %n_sample% ^
--eval_list %eval_list% ^
--model %model% ^
--batch_size_eval %batch_size_eval% ^
--best_model_dir %best_model_dir% ^
--cspn_norm_type %cspn_norm_type% ^
--cspn_step %cspn_step% ^
--resume_model_name %resume_model_name% ^
-n -r
