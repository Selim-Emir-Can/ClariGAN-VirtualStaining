import argparse
import os
import yaml
import copy
import torch
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
    "test_set_results.py",  # Placeholder for script name
    "--config", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\Template-LBBDM-f16_imagenetVQGAN.yaml",
    "--gpu_ids", "0",
    "--resume_model", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN\LBBDM-f16\checkpoint\top_model_epoch_98.pth",
    "--resume_optim", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN\LBBDM-f16\checkpoint\top_optim_sche_epoch_98.pth"
]

if __name__ == "__main__":
    nconfig = main()
    runner = get_runner(nconfig.runner, nconfig)
    bbdmnet = runner.initialize_model(nconfig)
    ckpt_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results\ClariGAN_imagenetVQGAN\LBBDM-f16\checkpoint\top_model_epoch_98.pth"
    bbdmnet.load_state_dict(torch.load(ckpt_path, weights_only=True)['model'])

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
    
    sample_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM_results_combined"
    runner.sample_to_eval_combined_with_uncertainty(bbdmnet, test_loader, sample_path=sample_path)