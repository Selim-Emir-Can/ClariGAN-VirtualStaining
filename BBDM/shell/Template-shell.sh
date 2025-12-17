#train
python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f8.yaml" --train --sample_at_start --save_top --gpu_ids 0

python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f8_sampleB.yaml" --train --sample_at_start --save_top --gpu_ids 0

python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f8_sampleB_OVERFIT.yaml" --train --sample_at_start --save_top --gpu_ids 0

python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN.yaml" --train --sample_at_start --save_top --gpu_ids 0

python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml" --train --sample_at_start --save_top --gpu_ids 0


python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml" --train --sample_at_start --save_top --gpu_ids 0


conda activate BBDM
cd "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM"
python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml" --train --sample_at_start --save_top --gpu_ids 0
python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_5percentBalanced.yaml" --train --sample_at_start --save_top --gpu_ids 0

python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml" --train --sample_at_start --save_top --gpu_ids 0 --resume_model C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_5percent\LBBDM-f16\checkpoint\latest_model_50.pth --resume_optim C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_5percent\LBBDM-f16\checkpoint\latest_optim_sche_50.pth




python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_BF.yaml" --train --sample_at_start --save_top --gpu_ids 0

python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_BFDF.yaml" --train --sample_at_start --save_top --gpu_ids 0 --resume_model "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BFDF_imagenetVQGAN_finetuned_50\LBBDM-f16\checkpoint\last_model.pth" --resume_optim "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BFDF_imagenetVQGAN_finetuned_50\LBBDM-f16\checkpoint\last_optim_sche.pth"


#test
python main.py --config "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f8.yaml" --sample_to_eval --gpu_ids 0 --resume_model "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN\LBBDM-f8\checkpoint\top_model_epoch_6.pth" --resume_optim "C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN\LBBDM-f8\checkpoint\top_optim_sche_epoch_6.pth"

#preprocess and evaluation
## rename
#python3 preprocess_and_evaluation.py -f rename_samples -r root/dir -s source/dir -t target/dir

## copy
#python3 preprocess_and_evaluation.py -f copy_samples -r root/dir -s source/dir -t target/dir

## LPIPS
#python3 preprocess_and_evaluation.py -f LPIPS -s source/dir -t target/dir -n 1

## max_min_LPIPS
#python3 preprocess_and_evaluation.py -f max_min_LPIPS -s source/dir -t target/dir -n 1

## diversity
#python3 preprocess_and_evaluation.py -f diversity -s source/dir -n 1

## fidelity
#fidelity --gpu 0 --fid --input1 path1 --input2 path2