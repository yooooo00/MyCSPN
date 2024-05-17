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
set best_model_dir=output\sgd0517_step12_mynet_inputrgb255_layer1_blurdepth_refinesparse_nomask
set refine_model_name=adam0513_train_refine
set refine_model_dir=output\%refine_model_name%
set cspn_norm_type=8sum
set cspn_step=12
set resume_model_name=best_train_36.97.pth
set resume_refine_model_name=best_model.pth

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
--refine_model_dir %refine_model_dir% ^
--resume_refine_model_name %resume_refine_model_name% ^
-n -r
