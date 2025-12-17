conda activate taming
cd C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,

python main.py --base configs/custom_vqganAll.yaml -t True --gpus 0,


# training
conda activate taming
cd C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers
python main.py --base C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\configs\ChrisBF.yaml --name CHRISBF_finetune --resume_from_checkpoint C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\pretrained_weights\last.ckpt -t True --gpus 0,


# discard
python C:\Users\ammic\Desktop\ClariGAN-DL\filter_dataset.py --folder_a C:\Users\ammic\Desktop\R1_training --folder_b C:\Users\ammic\Desktop\R3_training --keep_log C:\Users\ammic\Desktop\ClariGAN-DL\keep.txt --review_log C:\Users\ammic\Desktop\ClariGAN-DL\review.txt

python C:\Users\ammic\Desktop\ClariGAN-DL\filter_dataset.py --folder_a C:\Users\ammic\Desktop\R1_training --folder_b C:\Users\ammic\Desktop\R3_training --keep_log C:\Users\ammic\Desktop\ClariGAN-DL\keep.txt --review_log C:\Users\ammic\Desktop\ClariGAN-DL\review.txt --resume_file C:\Users\ammic\Desktop\ClariGAN-DL\review.txt