@echo off
SET CUDA_VISIBLE_DEVICES=0 | python hltrainval.py .
--data-dir ./data/maize_tassels_counting_uav_dataset .
--dataset uav .
--model tasselnetv2plus .
--exp tasselnetv2plus_rf110_i64o8_r0125_crop256_lr-2_bs9_epoch500 .
--data-list ./data/maize_tassels_counting_uav_dataset/train.txt .
--data-val-list ./data/maize_tassels_counting_uav_dataset/val.txt .
--restore-from model_best.pth.tar .
--image-mean 0.3859 0.4905 0.2895 .
--image-std 0.1718 0.1712 0.1518 .
--input-size 64 .
--output-stride 8 .
--resize-ratio 0.125 .
--optimizer sgd .
--milestones 200 400 .
--batch-size 9 .
--crop-size 256 256 .
--learning-rate 1e-2 .
--num-epochs 500 .
--num-workers 0 .
--print-every 10 .
--random-seed 2020 .
--val-every 10 .
--evaluate-only .
--save-output