CUDA_VISIBLE_DEVICES=0 python hltrainval.py \
--data-dir ./data/sorghum_head_counting_dataset \
--dataset shc \
--model tasselnetv2plus \
--exp tasselnetv2plus_dataset1_rf110_i64o8_r1_crop1024_lr-2_bs10_epoch500 \
--data-list ./data/sorghum_head_counting_dataset/dataset1_train.txt \
--data-val-list ./data/sorghum_head_counting_dataset/dataset1_test.txt \
--restore-from model_best.pth.tar \
--image-mean 0.3714 0.3609 0.2386 \
--image-std 0.2705 0.2567 0.2161 \
--input-size 64 \
--output-stride 8 \
--resize-ratio 1 \
--optimizer sgd \
--milestones 200 400 \
--batch-size 5 \
--crop-size 256 1024 \
--learning-rate 1e-2 \
--num-epochs 500 \
--num-workers 0 \
--print-every 5 \
--random-seed 2020 \
--val-every 10 \
--evaluate-only \
--save-output