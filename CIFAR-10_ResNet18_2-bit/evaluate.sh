############ directory to save result #############
mkdir ./save
mkdir ./dataset
model=resnet18_quant_tanh
dataset=cifar10
epochs=10
batch_size=20
optimizer=SGD
group_ch=16
wbit=2
abit=2
mode=mean
k=2
ratio=0.7
wd=0.0005
lr=0.1
#q_file=$4
save_path="./save/resnet18_quant_tanh/resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02/"
log_file="resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02.log"
pretrained_model="./save/resnet18_quant_tanh/resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02/model_best.pth.tar"
CUDA_LAUNCH_BLOCKING=1 python -W ignore train.py --dataset cifar10 --data_path ./dataset/     --model resnet18_quant_tanh     --save_path "./save/resnet18_quant_tanh/resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02/"     --epochs ${epochs}     --log_file  "resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02.log"    --lr  ${lr}     --schedule 60 120     --gammas 0.1 0.1     --batch_size ${batch_size}     --ngpu 1     --wd ${wd}     --k ${k}     --group_ch ${group_ch}     --wbit ${wbit}     --abit ${abit} --quant_bound 1.0 --number_levels 2 --offset_noise 2.5 --bitline_noise 6 --sram_depth 256 --finetune
#--resume  "./save/resnet18_quant_tanh/resnet18_quant_tanh_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse_g02/model_best.pth.tar"    --fine_tune