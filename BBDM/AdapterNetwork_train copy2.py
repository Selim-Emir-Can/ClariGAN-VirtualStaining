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


def BBDMsetup():
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

sys.argv = [
    "AdpterNetwork_results.py",  # Placeholder for script name
    "--config", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_BF.yaml",
    "--gpu_ids", "-1",
    "--resume_model", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_model_epoch_40.pth", # r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN_v1\LBBDM-f16\checkpoint\top_model_epoch_56.pth",
    "--resume_optim", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_optim_sche_epoch_40.pth" # r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN_v1\LBBDM-f16\checkpoint\top_optim_sche_epoch_56.pth"
]

if __name__ == "__main__":
    nconfig = BBDMsetup()

    nconfig.data.test.batch_size = 1
    nconfig.data.dataset_config.dataset_path = r"C:\Users\ammic\Downloads\BFDF-dataset"
    runner = get_runner(nconfig.runner, nconfig)
    runner.config.testing.sample_num = 2

    bbdmnet = runner.initialize_model(nconfig)
    ckpt_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_model_epoch_40.pth"
    bbdmnet.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu')['model']) # nconfig.training.device[0]

    train_dataset, val_dataset, test_dataset = get_dataset(nconfig.data)
    
    import torch
    import os
    import gc
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader


    def setup_dataloaders(train_dataset, val_dataset, test_dataset, config):
        """Create and return dataloaders for training, validation, and testing."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.train.batch_size,
            shuffle=config.data.train.shuffle,
            num_workers=8,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.val.batch_size,
            shuffle=config.data.val.shuffle,
            num_workers=8,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.test.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=True
        )
        
        return train_loader, val_loader, test_loader

    def setup_model(device):
        """Initialize and return the model."""
        # You can choose which model to use
        # adapter = AdapterNetwork(
        #     input_channels=3,  # For RGB images
        #     output_channels=3  # Assuming RGB output
        # )
        
        adapter = DeepLabV3PlusAdapter(
            input_channels=3,
            output_channels=3,
            encoder_name="resnet18",
            encoder_weights="imagenet"
        )
        
        adapter.to(device)
        print(f"Model loaded to device: {device}")
        
        return adapter

    def setup_training(adapter):
        """Setup and return loss function and optimizer."""
        criterion = create_combined_microscopy_loss()  # Could also use torch.nn.MSELoss()
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
        
        return criterion, optimizer

    def process_batch(bbdmnet, x_cond, device, sample_num=1):
        """Process a batch through the BBDM network with memory optimization."""
        bbdmnet.to(device)
        bbdmnet.eval()
        
        if sample_num > 1:
            bbdm_outputs = []
            for _ in range(sample_num):
                with torch.no_grad():
                    result = bbdmnet.sample(x_cond, clip_denoised=runner.config.testing.clip_denoised).detach()
                    bbdm_outputs.append(result)
                    del result
                    
            adapter_input = torch.stack(bbdm_outputs)  # Shape: (sample_num, B, C, H, W)
            del bbdm_outputs
        else:
            with torch.no_grad():
                result = bbdmnet.sample(x_cond, clip_denoised=runner.config.testing.clip_denoised)
                adapter_input = result.detach()
                del result
        
        bbdmnet.cpu()
        torch.cuda.empty_cache()
        
        return adapter_input.unsqueeze(0) if sample_num == 1 else adapter_input

    def save_checkpoint(adapter, optimizer, epoch, batch_idx, loss, is_best, checkpoint_dir, images_dir=None, output_img=None, target_img=None):
        """Save model checkpoint and optionally save images."""
        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'adapter_checkpoint_e{epoch+1}_b{batch_idx+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
        # If it's the best model, save it separately
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, best_path)
            print(f"Saved best model with loss: {loss:.4f}")
        
        # Optionally save images
        if images_dir and output_img is not None and target_img is not None:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(output_img)
            plt.title(f'Output e{epoch+1}_b{batch_idx+1}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(target_img)
            plt.title(f'Ground Truth e{epoch+1}_b{batch_idx+1}')
            plt.axis('off')
            
            plt.savefig(os.path.join(images_dir, f'comparison_e{epoch+1}_b{batch_idx+1}.png'))
            plt.close()
            
            # Save individual images
            plt.imsave(os.path.join(images_dir, f'output_e{epoch+1}_b{batch_idx+1}.png'), output_img)
            plt.imsave(os.path.join(images_dir, f'gt_e{epoch+1}_b{batch_idx+1}.png'), target_img)

    def validate(adapter, val_loader, criterion, device, bbdmnet, sample_num=1, save_results=True, results_dir=None, epoch=0):
        """Run validation and return average validation loss."""
        adapter.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        # Create validation results directory if needed
        if save_results and results_dir:
            val_results_dir = os.path.join(results_dir, f"validation_results_epoch_{epoch+1}")
            os.makedirs(val_results_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(tqdm(val_loader, desc="Validation", smoothing=0.01)):
                torch.cuda.empty_cache()
                
                try:
                    # Unpack batch data
                    (x, _), (x_cond, _) = val_batch
                    del val_batch
                    
                    # Move input data to device
                    x = x.to(device)
                    x_cond = x_cond.to(device)
                    
                    # Process through BBDM
                    adapter_input = process_batch(bbdmnet, x_cond, device, sample_num)
                    
                    # Forward pass through adapter
                    adapter_output = adapter(adapter_input, x_cond)
                    del adapter_input
                    torch.cuda.empty_cache()
                    
                    # Compute loss
                    loss = criterion(adapter_output, x)
                    val_loss = loss.item()
                    
                    # Save validation results (every 10 batches to avoid too many images)
                    if save_results and results_dir and (batch_idx % 10 == 0 or batch_idx < 5):
                        with torch.no_grad():
                            pred_grid = get_image_grid(adapter_output.cpu(), grid_size=1, to_normal=True)
                            label_grid = get_image_grid(x.cpu(), grid_size=1, to_normal=True)
                            
                            # Save comparison image
                            plt.figure(figsize=(10, 5))
                            
                            plt.subplot(1, 2, 1)
                            plt.imshow(pred_grid)
                            plt.title(f'Val Output e{epoch+1}_b{batch_idx+1}')
                            plt.axis('off')
                            
                            plt.subplot(1, 2, 2)
                            plt.imshow(label_grid)
                            plt.title(f'Val Ground Truth e{epoch+1}_b{batch_idx+1}')
                            plt.axis('off')
                            
                            plt.savefig(os.path.join(val_results_dir, f'val_comparison_e{epoch+1}_b{batch_idx+1}.png'))
                            plt.close()
                            
                            # Save individual images
                            plt.imsave(os.path.join(val_results_dir, f'val_output_e{epoch+1}_b{batch_idx+1}.png'), pred_grid)
                            plt.imsave(os.path.join(val_results_dir, f'val_gt_e{epoch+1}_b{batch_idx+1}.png'), label_grid)
                    
                    # Free memory
                    del x, x_cond, loss
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Update metrics
                    total_val_loss += val_loss
                    val_batch_count += 1
                    
                except RuntimeError as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
        
        # Return average validation loss
        return total_val_loss / max(1, val_batch_count)
    
    def train_epoch(adapter, train_loader, criterion, optimizer, device, bbdmnet, epoch, 
                    sample_num, checkpoint_dir, images_dir, checkpoint_freq=200):
        """Train for one epoch and return average loss."""
        adapter.train()
        pbar = tqdm(train_loader, smoothing=0.01)
        total_loss = 0
        batch_count = 0
        
        for batch_idx, train_batch in enumerate(pbar):
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            try:
                # Unpack batch data
                (x, _), (x_cond, _) = train_batch
                del train_batch
                
                # Move input data to device
                x = x.to(device)
                x_cond = x_cond.to(device)
                
                # Process through BBDM
                adapter_input = process_batch(bbdmnet, x_cond, device, sample_num)
                
                # Forward pass through adapter
                adapter_output = adapter(adapter_input, x_cond)
                del adapter_input
                torch.cuda.empty_cache()
                
                # Compute loss
                loss = criterion(adapter_output, x)
                current_loss = loss.item()
                
                # Create images for checkpointing if needed
                if (batch_idx + 1) % checkpoint_freq == 0:
                    with torch.no_grad():
                        pred_grid = get_image_grid(adapter_output.cpu(), grid_size=1, to_normal=True)
                        label_grid = get_image_grid(x.cpu(), grid_size=1, to_normal=True)
                
                # Free memory before backward
                adapter_output = adapter_output.cpu()
                del x, x_cond
                torch.cuda.empty_cache()
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Checkpointing
                if (batch_idx + 1) % checkpoint_freq == 0:
                    save_checkpoint(
                        adapter, optimizer, epoch, batch_idx, current_loss, 
                        is_best=False, 
                        checkpoint_dir=checkpoint_dir, 
                        images_dir=images_dir,
                        output_img=pred_grid,
                        target_img=label_grid
                    )
                
                # Update metrics
                total_loss += current_loss
                batch_count += 1
                
                # Cleanup
                del loss, adapter_output
                gc.collect()
                torch.cuda.empty_cache()
                
                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {current_loss:.4f}")
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Emergency cleanup
                for param in adapter.parameters():
                    if param.grad is not None:
                        param.grad = None
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Return average loss for the epoch
        return total_loss / max(1, batch_count)

    def train_model(train_loader, val_loader, adapter, criterion, optimizer, device, bbdmnet, 
                num_epochs=100, sample_num=1, checkpoint_dir="./checkpoints", 
                images_dir="./images", val_results_dir="./validation_results"):
        """Full training loop with validation and best model saving."""
        # Set environment variable to help with memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(val_results_dir, exist_ok=True)
        
        # Initialize variables to track best model
        best_val_loss = float('inf')
        
        # Prepare BBDM
        bbdmnet.to('cpu')
        bbdmnet.eval()
        
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = train_epoch(
                adapter, train_loader, criterion, optimizer, device, bbdmnet,
                epoch, sample_num, checkpoint_dir, images_dir
            )
            
            print(f"Epoch {epoch+1} completed with average training loss: {train_loss:.4f}")
            
            # Validate model
            print("Running validation...")
            val_loss = validate(
                adapter, val_loader, criterion, device, bbdmnet, 
                sample_num=sample_num, 
                save_results=True, 
                results_dir=val_results_dir, 
                epoch=epoch
            )
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save model if it's the best so far
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")
            
                # Save epoch checkpoint
                save_checkpoint(
                    adapter, optimizer, epoch, len(train_loader)-1, train_loss, 
                    is_best=is_best,
                    checkpoint_dir=checkpoint_dir
                )
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return adapter

    def main():
        """Main function to run the training pipeline."""
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup dataloaders
        train_loader, val_loader, _ = setup_dataloaders(
            train_dataset, val_dataset, test_dataset, nconfig
        )
        
        # Setup model
        adapter = setup_model(device)
        
        # Setup training components
        criterion, optimizer = setup_training(adapter)
        
        # Define training parameters
        num_epochs = 100
        sample_num = 1
        base_dir = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Weights"
        checkpoint_dir = base_dir
        images_dir = os.path.join(base_dir, "train_images")
        val_results_dir = os.path.join(base_dir, "validation_results")
        
        # Train the model
        trained_adapter = train_model(
            train_loader, val_loader, adapter, criterion, optimizer, device, bbdmnet,
            num_epochs=num_epochs, sample_num=sample_num,
            checkpoint_dir=checkpoint_dir, images_dir=images_dir, val_results_dir=val_results_dir
        )
        
        return trained_adapter
    
    main()

