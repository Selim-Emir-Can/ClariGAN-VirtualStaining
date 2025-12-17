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
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def CPU_singleGPU_launcher(config, train_set, val_set, test_set, save_name):
    set_random_seed(config.args.seed)

    ################
    config.data.dataset_name = config.data.dataset_name + '_' + save_name

    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train(train_set, val_set, test_set)
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


def main(train_set, val_set, test_set, save_name):
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
            CPU_singleGPU_launcher(nconfig, train_set, val_set, test_set, save_name)
    return nconfig

import os
import re
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
from collections import Counter

import os
import re
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split

def stratified_kfold_85_5_10(train_dir="train", k=10, seed=42):
    input_dir = os.path.join(train_dir, 'A')
    gt_dir = os.path.join(train_dir, 'B')

    r1_files = [f for f in os.listdir(input_dir) if ('BF' in f) and os.path.isfile(os.path.join(input_dir, f))]

    data = []
    pattern = re.compile(r'^(Skin[A-Z]\d|[A-Z]\d)_(i{1,3})_(BF|Stained)_r\d+_c\d+\.tif$', re.IGNORECASE)

    # First pass: parse filenames and count labels
    for filename in r1_files:
        match = pattern.search(filename)
        if not match:
            print("Skipping unmatched file:", filename)
            continue
        letter = match.group(1)
        number = match.group(2)
        original_label = f"{letter}_{number}"
        r3_filename = filename.replace("BF", "Stained", 1)
        r1_path = os.path.join(input_dir, filename)
        r3_path = os.path.join(gt_dir, r3_filename)

        if os.path.exists(r3_path):
            data.append({
                "r1": r1_path,
                "r3": r3_path,
                "letter": letter,
                "number": number,
                "original_label": original_label
            })

    # Count classes
    original_labels = [item["original_label"] for item in data]
    label_counts = Counter(original_labels)

    # Merge small classes (only fallback strategy here)
    for item in data:
        label = item["original_label"]
        if label_counts[label] < k:
            # Simplistic merge strategy: drop repetition info if needed
            fallback_label = item["letter"]
            item["label"] = fallback_label
        else:
            item["label"] = label

    # Final label list after merging
    final_labels = [item["label"] for item in data]
    pairs = [(item["r1"], item["r3"]) for item in data]

    # Final check (optional safety net)
    final_counts = Counter(final_labels)
    too_small = {lbl: c for lbl, c in final_counts.items() if c < k}
    if too_small:
        raise ValueError(
            f"After merging, the following classes still have fewer than {k} samples:\n" +
            "\n".join([f"{lbl}: {c}" for lbl, c in too_small.items()])
        )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    all_indices = list(skf.split(pairs, final_labels))

    # Split folds
    folds = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(all_indices):
        trainval_data = [pairs[i] for i in trainval_idx]
        trainval_labels = [final_labels[i] for i in trainval_idx]
        test_data = [pairs[i] for i in test_idx]

        val_ratio = 1 / 18  # ~5% of total
        train_data, val_data, _, _ = train_test_split(
            trainval_data, trainval_labels,
            test_size=val_ratio,
            stratify=trainval_labels,
            random_state=seed
        )

        folds.append({
            "fold": fold_idx,
            "train": train_data,
            "val": val_data,
            "test": test_data
        })

    return folds


from collections import Counter
import re

# Extract labels again from filenames in splits
def get_labels_from_split(split):
    labels = []
    # Match 'SkinB1', 'B1', 'SkinA2', etc., at the start of the filename
    pattern = re.compile(r'^(Skin[A-Z]\d|[A-Z]\d)_(i{1,3})', re.IGNORECASE)
    for r1_path, _ in split:
        filename = os.path.basename(r1_path)
        match = pattern.match(filename)
        if match:
            letter = match.group(1)
            number = match.group(2)
            labels.append(f"{letter}_{number}")
    return labels

from tqdm import tqdm

# Generate the folds (85/5/10 stratified)
if __name__ == "__main__":
    splits = stratified_kfold_85_5_10(r"C:\Users\ammic\Downloads\BF-dataset\train", k=10) # change the path to BBDM dataset train directory

    print("Number of Train R1 images:", len(splits[0]['train']))
    print("Number of Val R1 images:", len(splits[0]['val']))
    print("Number of Test R1 images:", len(splits[0]['test']))

    # For fold 0
    train_labels = get_labels_from_split(splits[0]['train'])
    val_labels = get_labels_from_split(splits[0]['val'])
    test_labels = get_labels_from_split(splits[0]['test'])

    
    print("Number of Train R1 images:", len(train_labels))
    print("Train label counts:", Counter(train_labels))

    print("Number of Val R1 images:", len(val_labels))
    print("Val label counts:", Counter(val_labels))

    print("Number of Test R1 images:", len(test_labels))
    print("Test label counts:", Counter(test_labels))

    counter = 0 

    sys.argv = [
        "k-fold_validation.py",  # Placeholder for script name
        "--config", r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\configs\ChrisML.yaml", # change it to your config file path
        "--train",
        "--sample_at_start",
        "--save_top",
        "--gpu_ids", "0"]

    # Loop through each fold
    for fold in splits:
        # === Data Loading (Customize Here) ===
        fold_num = fold['fold']
        train_set = fold['train']
        val_set = fold['val']
        test_set = fold['test']

        # print(f"\n=== Fold {fold_num} ===")
        # print(f"Train size: {len(train_set)}")
        # print(f"Val size:   {len(val_set)}")
        # print(f"Test size:  {len(test_set)}")

        

        save_name = f'fold_{counter}'

        # === Model Initialization ===
        # === Training Loop ===

        nconfig = main(train_set, val_set, test_set, save_name)

        # === Evaluation ===
        nconfig.data.test.batch_size = 1
        nconfig.data.dataset_config.dataset_path = r"C:\Users\ammic\Downloads\BF-dataset\train" # change the path to BBDM dataset train directory
        runner = get_runner(nconfig.runner, nconfig)
        bbdmnet = runner.initialize_model(nconfig)

        import os
        import glob
        import re

        def find_latest_ckpt(base_path, dataset_name):
            ckpt_dir = os.path.join(base_path, dataset_name, "LBBDM-f16", "checkpoint")
            pattern = os.path.join(ckpt_dir, "top_model_epoch_*.pth")
            checkpoint_files = glob.glob(pattern)

            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

            # Extract epoch numbers and sort
            def extract_epoch(fname):
                match = re.search(r"epoch_(\d+)", fname)
                return int(match.group(1)) if match else -1

            checkpoint_files.sort(key=extract_epoch, reverse=True)
            return checkpoint_files[0]  # return the latest one
        
        base_path = r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results" # WHERE YOU HAVE YOUR MODEL WEIGHTS change this to where you want to save your results
        dataset_name = nconfig.data.dataset_name
        ckpt_path = find_latest_ckpt(base_path, dataset_name)
        # ckpt_path = os.path.join(r"C:\Users\ammic\Desktop\ClariGAN-DL\BBDM\results",nconfig.data.dataset_name,"LBBDM-f16\checkpoint") + r"top_model_epoch_40.pth" 

        bbdmnet.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=nconfig.training.device[0])['model'])

        nconfig.data.dataset_type = 'custom_aligned'

        _, _, test_dataset = get_dataset(nconfig.data, train_set, val_set, test_set)

        test_loader = DataLoader(test_dataset,
                                    batch_size=nconfig.data.test.batch_size,
                                    shuffle=False,
                                    num_workers=8,
                                    drop_last=True)
        
        print('test size: ', len(test_loader))

        sample_path = os.path.join(r"C:\Users\ammic\Desktop\ClariGAN-DL\CHRISk-fold_results", save_name) # WHERE YOU HAVE TEST RESULTS FOR EACH FOLD of the k-fold cross validation
        runner.sample_to_eval_combined_with_uncertainty(bbdmnet, test_loader, sample_path=sample_path)

        # === Data Loading (Customize Here) ===
        # train_loader = load_dataset(train_set)
        # val_loader = load_dataset(val_set)
        # test_loader = load_dataset(test_set)

        # === Model Initialization ===
        # model = init_model()

        # === Training Loop ===
        # train_model(model, train_loader, val_loader)

        # === Evaluation ===
        # metrics = evaluate_model(model, test_loader)
        # print(f"Fold {fold_num} Test Accuracy: {metrics['accuracy']:.2f}%")

        # === Optional: Save results per fold ===
        # save_results(metrics, fold_num)
        counter = counter + 1
