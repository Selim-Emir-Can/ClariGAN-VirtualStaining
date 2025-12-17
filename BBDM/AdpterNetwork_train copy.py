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

sys.argv = [
    "AdpterNetwork_results.py",  # Placeholder for script name
    "--config", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN_BF.yaml",
    "--gpu_ids", "-1",
    "--resume_model", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_model_epoch_40.pth", # r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN_v1\LBBDM-f16\checkpoint\top_model_epoch_56.pth",
    "--resume_optim", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_optim_sche_epoch_40.pth" # r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN_v1\LBBDM-f16\checkpoint\top_optim_sche_epoch_56.pth"
]

if __name__ == "__main__":
    nconfig = main()

    nconfig.data.test.batch_size = 1
    nconfig.data.dataset_config.dataset_path = r"C:\Users\ammic\Downloads\BFDF-dataset"
    runner = get_runner(nconfig.runner, nconfig)
    runner.config.testing.sample_num = 2

    bbdmnet = runner.initialize_model(nconfig)
    ckpt_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\BF_imagenetVQGAN_finetuned\LBBDM-f16\checkpoint\top_model_epoch_40.pth"
    bbdmnet.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu')['model']) # nconfig.training.device[0]

    train_dataset, val_dataset, test_dataset = get_dataset(nconfig.data)

    train_loader = DataLoader(train_dataset,
                                batch_size=nconfig.data.train.batch_size,
                                shuffle=nconfig.data.train.shuffle,
                                num_workers=8,
                                drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=nconfig.data.val.batch_size,
                            shuffle=nconfig.data.val.shuffle,
                            num_workers=8,
                            drop_last=True)

    test_loader = DataLoader(test_dataset,
                                batch_size=nconfig.data.test.batch_size,
                                shuffle=False,
                                num_workers=8,
                                drop_last=True)


    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the AdapterNetwork object
    input_channels = 3  # For RGB images
    output_channels = 3  # Assuming we want RGB output


    # Create an instance of the AdapterNetwork
    adapter = AdapterNetwork(
        input_channels=input_channels,
        output_channels=output_channels,
    )

    # Load the model onto the specified device (GPU/CPU)
    adapter.to(device)

    # Optionally, print the model summary or check if it's on the correct device
    # print(adapter)
    print(f"Model loaded to device: {device}")

    # Define the loss function and optimizer

    criterion = create_combined_microscopy_loss() # torch.nn.MSELoss()  # Use CrossEntropyLoss for classification tasks
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)

    # Adapter training loop with memory leak fix
    num_epochs = 10  # Define the number of epochs
    sample_num = 1
    bbdmnet.to('cpu')  # Keep model on CPU by default
    bbdmnet.eval()     # Ensure eval mode

    # Set environment variable to help with memory fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Track metrics
    print(f"Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Iterate over the test batches
        pbar = tqdm(test_loader, smoothing=0.01)
        total_loss = 0
        batch_count = 0
        
        for batch_idx, test_batch in enumerate(pbar):
            # Completely clear GPU memory at start of each batch
            torch.cuda.empty_cache()
            
            try:
                # Unpack batch data
                (x, _), (x_cond, _) = test_batch
                del test_batch
                
                # Move input data to device
                x = x.to(device)
                x_cond = x_cond.to(device)

                if(sample_num > 1):
                    bbdm_outputs = []
                    for j in range(sample_num):
                        with torch.no_grad():
                            # Move model to GPU for inference
                            bbdmnet.to(device)
                            result = bbdmnet.sample(x_cond, clip_denoised=runner.config.testing.clip_denoised).detach()
                            bbdmnet.cpu()

                            # Move to CPU and append to the list
                            bbdm_outputs.append(result)
                            del result

                    # Convert the list of outputs into a tensor
                    adapter_input = torch.stack(bbdm_outputs)  # Shape becomes (sample_num, B, C, H, W)

                    # Free GPU memory used by bbdmnet
                    del bbdm_outputs
                    torch.cuda.empty_cache()
                else:
                
                    # IMPORTANT: Process only ONE sample at a time to reduce memory use
                    with torch.no_grad():
                        # Move model to GPU for inference
                        bbdmnet.to(device)
                        result = bbdmnet.sample(x_cond, clip_denoised=runner.config.testing.clip_denoised)
                        # Store result and immediately move model back to CPU
                        adapter_input = result.detach()
                        del result
                        bbdmnet.cpu()
                        torch.cuda.empty_cache()
                
                # Forward pass through adapter
                adapter_output = adapter(adapter_input.unsqueeze(0), x_cond) # shape (B,3,H,W)
                del adapter_input
                torch.cuda.empty_cache()
                
                # Compute loss
                loss = criterion(adapter_output, x)
                current_loss = loss.item()

                # with torch.no_grad():
                #     plt.imshow(get_image_grid(adapter_output, grid_size=1, to_normal=True))
                #     plt.title('output')
                #     plt.show()

                #     plt.imshow(get_image_grid(x, grid_size=1, to_normal=True))
                #     plt.title('gt')
                #     plt.show()


                # Free memory before backward
                adapter_output = adapter_output.to('cpu')
                x = x.to('cpu')
                torch.cuda.empty_cache()

                
                # Zero gradients, compute backward, step optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Add regular checkpointing to avoid losing all progress
                if (batch_idx + 1) % 100 == 0:
                    torch.save(adapter.state_dict(), os.path.join(r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Weights", f'adapter_checkpoint_e{epoch+1}_b{batch_idx+1}.pth'))

                    with torch.no_grad():
                        pred_grid = get_image_grid(adapter_output, grid_size=1, to_normal=True)
                        plt.imshow(pred_grid)
                        plt.title(f'output epoch{epoch+1}_batch{batch_idx+1}')
                        plt.imsave(os.path.join(r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Weights\images", f'output_e{epoch+1}_b{batch_idx+1}.png'), pred_grid)

                        label_grid = get_image_grid(x, grid_size=1, to_normal=True)
                        plt.imshow(label_grid)
                        plt.title(f'gt epoch{epoch+1}_batch{batch_idx+1}')
                        plt.imsave(os.path.join(r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Weights\images", f'gt_e{epoch+1}_b{batch_idx+1}.png'), label_grid)
                
                del adapter_output, x

                
                # Update metrics
                total_loss += current_loss
                batch_count += 1
                
                # Extensive cleanup after each batch
                del loss, x_cond
                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Update progress bar
                pbar.set_description(f"Batch {batch_idx+1}, Loss: {current_loss:.4f}")
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                # Emergency cleanup
                for param in adapter.parameters():
                    if param.grad is not None:
                        param.grad = None
                torch.cuda.empty_cache()
                gc.collect()
                # Skip to next batch
                continue
            
            
        # End of epoch reporting
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1} completed with average loss: {avg_loss:.4f}")
            # Save model after each epoch
            torch.save(adapter.state_dict(), os.path.join(r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\AdapterNetwork_Weights", f"adapter_model_epoch_{epoch+1}.pth"))

    print("Training completed!")


    # num_epochs = 10  # Define the number of epochs
    # bbdmnet.to('cpu')
    # bbdmnet.eval()

    # for epoch in tqdm(range(num_epochs)):
    #     # Use tqdm for progress bar
    #     pbar = tqdm(test_loader, total=len(train_loader), smoothing=0.01)
    #     batch_size = runner.config.data.test.batch_size
    #     sample_num = runner.config.testing.sample_num

    #     # Iterate over the test batches
    #     for test_batch in pbar:
    #         torch.cuda.empty_cache()
    #         with torch.no_grad():
    #             (x, _), (x_cond, _) = test_batch
    #             del test_batch

    #             # Move data to device
    #             x = x.to(device)
    #             x_cond = x_cond.to(device)

    #             bbdm_outputs = []

    #             # Generate samples using the `bbdmnet` model
    #             for j in range(sample_num):
    #                 # Assuming `bbdmnet.sample` returns outputs of shape (B, C, H, W)
    #                 bbdmnet.to(device)
    #                 result = bbdmnet.sample(x_cond, clip_denoised=runner.config.testing.clip_denoised).detach().cpu()
    #                 bbdmnet.cpu()

    #                 # Move to CPU and append to the list
    #                 bbdm_outputs.append(result)

    #             # Convert the list of outputs into a tensor
    #             adapter_input = torch.stack(bbdm_outputs)  # Shape becomes (sample_num, B, C, H, W)

    #             # Free GPU memory used by bbdmnet
    #             del bbdm_outputs
    #             torch.cuda.empty_cache()


    #         # Now, pass the output into the Adapter Network
    #         adapter_output = adapter(adapter_input.to(device), x_cond)
    #         del adapter_input  # Free memory


    #         # Compute the loss
    #         loss = criterion(adapter_output, x)  # Assuming `x_cond` is the target ground truth
    #         del x_cond, x, adapter_output # Free memory

    #         # Zero the gradients
    #         optimizer.zero_grad()

    #         # Backpropagate and optimize
    #         loss.backward()
    #         optimizer.step()

    #         # Update progress bar
    #         with torch.no_grad():
    #             pbar.set_description(f"Loss: {loss.item():.4f}")
    #         del loss  # Free memory
    #         optimizer.zero_grad()
    #         bbdmnet.cpu()

    #     # # Optionally, save the model after every epoch
    #     # torch.save(adapter.state_dict(), f"adapter_model_epoch_{epoch+1}.pth")
    #     # print(f"Epoch {epoch+1} complete. Model saved.")

    #     # Optionally, print the loss at the end of the epoch
    #     # print(f"Epoch {epoch+1} completed with loss: {loss.item():.4f}")
