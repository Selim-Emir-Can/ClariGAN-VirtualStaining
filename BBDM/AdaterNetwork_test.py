import argparse
import os
import yaml
import copy
import torch
from AdapterNetwork import *
from AdapterNetwork_Losses import *
import random
import numpy as np

from utils import dict2namespace, get_runner, namespace2dict
import torch.multiprocessing as mp
import torch.distributed as dist

import sys

from runners.DiffusionBasedModelRunners import BBDMRunner
# from model.VQGAN.taming.data.custom import CustomTest, CustomTestClariGAN
from datasets.custom import CustomAlignedDataset
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid, save_image

@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid

def run_test_inference():
    # Set path to saved adapter model
    adapter_model_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Weights\adapter_model_epoch_10.pth"
    # Set path to BBDM model
    bbdm_model_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_model_epoch_40.pth"
    # Set path for saving test results
    output_dir = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Results\test_inference"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config and set up test dataset
    from main import parse_args_and_config
    import sys
    
    # Set arguments for loading config
    sys.argv = [
        "AdpterNetwork_results.py",
        "--config", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_BF.yaml",
        "--gpu_ids", "-1",
        "--resume_model", bbdm_model_path,
        "--resume_optim", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_optim_sche_epoch_40.pth"
    ]
    
    # Get config
    nconfig = main()
    nconfig.data.test.batch_size = 1
    nconfig.data.dataset_config.dataset_path = r"C:\Users\ammic\Downloads\BFDF-dataset"
    
    # Set up runners and models
    from utils import get_runner
    
    # Initialize the BBDM model
    runner = get_runner(nconfig.runner, nconfig)
    bbdmnet = runner.initialize_model(nconfig)
    bbdmnet.load_state_dict(torch.load(bbdm_model_path, weights_only=True, map_location='cpu')['model'])
    bbdmnet.eval()
    
    # Get datasets
    _, _, test_dataset = get_dataset(nconfig.data)
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=nconfig.data.test.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize adapter network
    adapter = AdapterNetwork(input_channels=3, output_channels=3)
    adapter.load_state_dict(torch.load(adapter_model_path, map_location='cpu'))
    adapter.to(device)
    adapter.eval()
    
    # Create a text file to log metrics
    metrics_file = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Test Inference Metrics\n")
        f.write("======================\n\n")
    
    # Import metrics
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import numpy as np
    
    # Prepare to track metrics
    total_bbdm_ssim = 0
    total_bbdm_psnr = 0
    total_adapter_ssim = 0
    total_adapter_psnr = 0
    sample_count = 0
    
    print("Starting test inference...")
    
    # Process test samples
    for batch_idx, test_batch in enumerate(tqdm(test_loader)):
        # Clear CUDA cache before processing each batch
        torch.cuda.empty_cache()
        
        try:
            # Unpack batch data
            (x, _), (x_cond, _) = test_batch
            
            # Move input data to device
            x = x.to(device)
            x_cond = x_cond.to(device)
            
            # Generate output from BBDM model
            with torch.no_grad():
                # Move BBDM to device, generate output, then move back to CPU
                bbdmnet.to(device)
                bbdm_output = bbdmnet.sample(x_cond, clip_denoised=runner.config.testing.clip_denoised).detach()
                bbdmnet.cpu()
                
                # Process through adapter
                adapter_output = adapter(bbdm_output.unsqueeze(0), x_cond)
                
                # Save images
                sample_dir = os.path.join(output_dir, f"sample_{batch_idx}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Convert tensors to image grids
                bbdm_grid = get_image_grid(bbdm_output, grid_size=1, to_normal=True)
                adapter_grid = get_image_grid(adapter_output, grid_size=1, to_normal=True)
                target_grid = get_image_grid(x, grid_size=1, to_normal=True)
                
                # Calculate metrics for BBDM output
                bbdm_np = bbdm_grid / 255.0
                adapter_np = adapter_grid / 255.0
                target_np = target_grid / 255.0
                
                # Calculate SSIM and PSNR for BBDM output
                bbdm_ssim = ssim(bbdm_np, target_np, channel_axis=2, data_range=1.0)
                bbdm_psnr = psnr(target_np, bbdm_np, data_range=1.0)
                
                # Calculate SSIM and PSNR for adapter output
                adapter_ssim = ssim(adapter_np, target_np, channel_axis=2, data_range=1.0)
                adapter_psnr = psnr(target_np, adapter_np, data_range=1.0)
                
                # Update totals
                total_bbdm_ssim += bbdm_ssim
                total_bbdm_psnr += bbdm_psnr
                total_adapter_ssim += adapter_ssim
                total_adapter_psnr += adapter_psnr
                sample_count += 1
                
                # Save images with metrics in the titles
                plt.figure(figsize=(8, 6))
                plt.imshow(bbdm_grid)
                plt.title(f"BBDM Output - SSIM: {bbdm_ssim:.4f}, PSNR: {bbdm_psnr:.2f} dB")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, "bbdm_output.png"), dpi=150)
                plt.close()
                
                plt.figure(figsize=(8, 6))
                plt.imshow(adapter_grid)
                plt.title(f"Adapter Output - SSIM: {adapter_ssim:.4f}, PSNR: {adapter_psnr:.2f} dB")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, "adapter_output.png"), dpi=150)
                plt.close()
                
                plt.figure(figsize=(8, 6))
                plt.imshow(target_grid)
                plt.title("Ground Truth")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, "ground_truth.png"), dpi=150)
                plt.close()
                
                # Save side-by-side comparison
                plt.figure(figsize=(18, 6))
                
                plt.subplot(1, 3, 1)
                plt.imshow(bbdm_grid)
                plt.title(f"BBDM Output\nSSIM: {bbdm_ssim:.4f}, PSNR: {bbdm_psnr:.2f} dB")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(adapter_grid)
                plt.title(f"Adapter Output\nSSIM: {adapter_ssim:.4f}, PSNR: {adapter_psnr:.2f} dB")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(target_grid)
                plt.title("Ground Truth")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, "comparison.png"), dpi=150)
                plt.close()
                
                # Write individual metrics to file
                with open(metrics_file, 'a') as f:
                    f.write(f"Sample {batch_idx}:\n")
                    f.write(f"  BBDM:    SSIM = {bbdm_ssim:.4f}, PSNR = {bbdm_psnr:.2f} dB\n")
                    f.write(f"  Adapter: SSIM = {adapter_ssim:.4f}, PSNR = {adapter_psnr:.2f} dB\n")
                    f.write(f"  Improvement: SSIM +{adapter_ssim-bbdm_ssim:.4f}, PSNR +{adapter_psnr-bbdm_psnr:.2f} dB\n\n")
                
                # Free memory
                del bbdm_output, adapter_output, x, x_cond
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Calculate and save average metrics
    if sample_count > 0:
        avg_bbdm_ssim = total_bbdm_ssim / sample_count
        avg_bbdm_psnr = total_bbdm_psnr / sample_count
        avg_adapter_ssim = total_adapter_ssim / sample_count
        avg_adapter_psnr = total_adapter_psnr / sample_count
        
        with open(metrics_file, 'a') as f:
            f.write(f"Average Metrics for {sample_count} samples:\n")
            f.write(f"  BBDM:    Avg SSIM = {avg_bbdm_ssim:.4f}, Avg PSNR = {avg_bbdm_psnr:.2f} dB\n")
            f.write(f"  Adapter: Avg SSIM = {avg_adapter_ssim:.4f}, Avg PSNR = {avg_adapter_psnr:.2f} dB\n")
            f.write(f"  Avg Improvement: SSIM +{avg_adapter_ssim-avg_bbdm_ssim:.4f}, PSNR +{avg_adapter_psnr-avg_bbdm_psnr:.2f} dB\n")
        
        print(f"Inference complete. Processed {sample_count} samples.")
        print(f"BBDM: Avg SSIM = {avg_bbdm_ssim:.4f}, Avg PSNR = {avg_bbdm_psnr:.2f} dB")
        print(f"Adapter: Avg SSIM = {avg_adapter_ssim:.4f}, Avg PSNR = {avg_adapter_psnr:.2f} dB")
        print(f"Average Improvement: SSIM +{avg_adapter_ssim-avg_bbdm_ssim:.4f}, PSNR +{avg_adapter_psnr-avg_bbdm_psnr:.2f} dB")
        
        # Create summary visualization with average metrics
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(2)
        
        ssim_vals = [avg_bbdm_ssim, avg_adapter_ssim]
        psnr_vals = [avg_bbdm_psnr, avg_adapter_psnr]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(index, ssim_vals, bar_width, label='SSIM (higher is better)')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('SSIM Value')
        ax1.set_title('Average SSIM Comparison')
        ax1.set_xticks(index)
        ax1.set_xticklabels(['BBDM', 'Adapter'])
        for i, v in enumerate(ssim_vals):
            ax1.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        ax2.bar(index, psnr_vals, bar_width, color='orange', label='PSNR (higher is better)')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Average PSNR Comparison')
        ax2.set_xticks(index)
        ax2.set_xticklabels(['BBDM', 'Adapter'])
        for i, v in enumerate(psnr_vals):
            ax2.text(i, v + 0.5, f"{v:.2f} dB", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_summary.png"), dpi=150)
        plt.close()
    else:
        print("No samples were successfully processed.")


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def DDP_run_fn(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    set_random_seed(config.args.seed)

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    config.training.device = [torch.device("cuda:%d" % local_rank)]
    print('using device:', config.training.device)
    config.training.local_rank = local_rank
    return # STOP FROM DEFINING RUNNER INSTANCE

    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    return # STOP FROM DEFINING RUNNER INSTANCE
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def DDP_launcher(world_size, run_fn, config):
    raise Exception("Not Allowing Multiple GPU Inference")
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)

def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args

    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)
    return nconfig

if __name__ == "__main__":
    run_test_inference()