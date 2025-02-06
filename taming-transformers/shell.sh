conda activate taming
cd C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,

python main.py --base configs/custom_vqganAll.yaml -t True --gpus 0,


# training
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,

# fine-tuning
python main.py --base C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\configs\custom_vqgan_f16_imagenet.yaml --name vqgan_imagenet_f16_16384_finetuned-epoch22 --resume_from_checkpoint C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\logs\2025-01-04T21-10-47_vqgan_imagenet_f16_16384_finetuned-microscopy\checkpoints\epoch=000022-val_rec_loss-val_loss=0.0000.ckpt -t True --gpus 0,

python main.py --base C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\configs\custom_vqgan_f16_imagenet.yaml --name vqgan_imagenet_f16_16384_finetuned-full --resume_from_checkpoint C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\pretrained_weights\last.ckpt -t True --gpus 0,


python main.py --base C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\configs\custom_vqgan_f16_imagenet.yaml --name vqgan_imagenet_f16_16384_finetuned-full --resume_from_checkpoint C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\logs\2025-01-20T23-34-56_vqgan_imagenet_f16_16384_finetuned-full\checkpoints\last.ckpt -t True --gpus 0,


python main.py --base C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\configs\custom_vqgan_f16_imagenet.yaml --name clariGAN_finetune --resume_from_checkpoint C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\pretrained_weights\last.ckpt -t True --gpus 0,

python main.py --base C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\configs\custom_vqgan_f16_imagenet.yaml --name clariGAN_finetune --resume_from_checkpoint C:\Users\ammic\Desktop\ClariGAN-DL\taming-transformers\logs\2025-02-02T14-00-16_clariGAN_finetune\checkpoints\last.ckpt -t True --gpus 0,

# discard
python C:\Users\ammic\Desktop\ClariGAN-DL\filter_dataset.py --folder_a C:\Users\ammic\Desktop\R1_training --folder_b C:\Users\ammic\Desktop\R3_training --keep_log C:\Users\ammic\Desktop\ClariGAN-DL\keep.txt --review_log C:\Users\ammic\Desktop\ClariGAN-DL\review.txt

python C:\Users\ammic\Desktop\ClariGAN-DL\filter_dataset.py --folder_a C:\Users\ammic\Desktop\R1_training --folder_b C:\Users\ammic\Desktop\R3_training --keep_log C:\Users\ammic\Desktop\ClariGAN-DL\keep.txt --review_log C:\Users\ammic\Desktop\ClariGAN-DL\review.txt --resume_file C:\Users\ammic\Desktop\ClariGAN-DL\review.txt