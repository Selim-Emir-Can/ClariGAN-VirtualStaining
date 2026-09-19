"""Specimen-grouped k-fold driver for ClariDi (replaces k-fold_validation.py).

Every path is a CLI argument, and the split is produced by specimen_kfold.py, which
assigns whole physical specimens to train/val/test.  Nothing is hardcoded and no
specimen can straddle partitions (assert_no_leakage runs per fold before training).

Dry run (prints the fold table, trains nothing):
  python kfold_grouped.py --data_root /path/to/bbdm --dry_run

Train:
  python kfold_grouped.py --data_root /path/to/bbdm \
      --config configs/Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml \
      --vqgan_ckpt /path/to/vqgan.ckpt --results_root /path/to/results \
      --gpu_ids 0,1,2 --scheme loso
"""
import argparse
import copy
import glob
import os
import re
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from utils import dict2namespace, get_runner, namespace2dict
from runners.utils import get_dataset
from specimen_kfold import (load_manifest, make_folds, assert_no_leakage,
                            fold_table, format_fold_table, save_fold_assignments)

import main as bbdm_main  # reuse set_random_seed / DDP_run_fn plumbing

# Importing these registers them in Registers.runners; the configs select one by name.
# Without the imports, BBDMRunner_trainable_encoder / BBDMRunner_pixel_loss are missing
# and get_runner raises KeyError the moment those ablations start.
from runners.DiffusionBasedModelRunners import BBDMRunner_trainable_encoder  # noqa: F401
from runners.DiffusionBasedModelRunners import BBDMRunner_pixel_loss  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                   help="dataset dir holding train/A, train/B and manifest.csv")
    p.add_argument("--manifest", default=None, help="default <data_root>/manifest.csv")
    p.add_argument("--config", default="configs/Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml")
    p.add_argument("--vqgan_ckpt", default=None,
                   help="override model.VQGAN.params.ckpt_path (latent models only)")
    p.add_argument("--results_root", default=None,
                   help="where checkpoints/logs go; default <data_root>/../results")
    p.add_argument("--samples_root", default=None,
                   help="where per-fold test samples go; default <results_root>/../k-fold_samples")
    p.add_argument("--scheme", choices=["loso", "grouped"], default="loso")
    p.add_argument("--n_folds", type=int, default=5, help="only used with --scheme grouped")
    p.add_argument("--folds", default=None,
                   help="comma-separated fold indices to run (default: all)")
    p.add_argument("--gpu_ids", default="0", help="e.g. 0 or 0,1,2 (cpu=-1)")
    p.add_argument("--port", default="12355", help="DDP master port")
    p.add_argument("--seed", type=int, default=1234, help="training seed")
    p.add_argument("--sample_seed", type=int, default=1234,
                   help="sampling seed passed to eval_fold.py (gen j = sample_seed + j); kept at 1234 "
                        "for multi-seed training runs so they stay comparable")
    p.add_argument("--tag", default="specimen_grouped",
                   help="suffix used in dataset_name / sample dir names")
    p.add_argument("--accumulate_grad_batches", type=int, default=None,
                   help="override; default keeps effective batch = batch_size * accum of the config")
    p.add_argument("--max_epoch", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="print fold table + leakage check and exit without training")
    p.add_argument("--skip_train", action="store_true",
                   help="evaluate existing checkpoints only")
    p.add_argument("--deliverables_root", default=None,
                   help="where eval_fold.py writes samples/metrics; default <project>/deliverables")
    p.add_argument("--experiment", default=None,
                   help="experiment name for the deliverables (default: derived from --tag)")
    p.add_argument("--eval_max_tiles", type=int, default=None,
                   help="probe only: stop evaluation after this many test tiles")
    p.add_argument("--legacy_eval", action="store_true",
                   help="use the old sample_to_eval_combined_with_uncertainty instead of eval_fold.py")
    p.add_argument("--keep_optim", action="store_true",
                   help="keep optimizer/last/latest checkpoints after a fold finishes "
                        "(default prunes to top_model_epoch_*.pth only: 13 GB -> 2.4 GB per fold)")
    return p.parse_args()


def build_config(a, save_name):
    """Load the YAML and apply every path/runtime override as a namespace config."""
    with open(a.config) as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2namespace(dict_config)

    # argparse-style namespace the runners expect on config.args
    cfg.args = argparse.Namespace(
        config=a.config, seed=a.seed, result_path=a.results_root, train=not a.skip_train,
        sample_to_eval=False, sample_at_start=True, save_top=True,
        gpu_ids=a.gpu_ids, port=a.port, resume_model=None, resume_optim=None,
        max_epoch=a.max_epoch, max_steps=a.max_steps)

    cfg.data.dataset_config.dataset_path = a.data_root
    if a.vqgan_ckpt is not None and hasattr(cfg.model, "VQGAN"):
        cfg.model.VQGAN.params.ckpt_path = a.vqgan_ckpt
    if a.max_epoch is not None:
        cfg.training.n_epochs = a.max_epoch
    if a.max_steps is not None:
        cfg.training.n_steps = a.max_steps

    # Keep the effective batch size (batch_size * accum * world_size) equal to the
    # original single-GPU setting by dividing accumulation across the GPUs.
    n_gpu = 1 if a.gpu_ids.strip() == "-1" else len([g for g in a.gpu_ids.split(",") if g != ""])
    if a.accumulate_grad_batches is not None:
        cfg.training.accumulate_grad_batches = a.accumulate_grad_batches
    elif n_gpu > 1:
        base = cfg.training.accumulate_grad_batches
        if base % n_gpu != 0:
            raise SystemExit(
                f"accumulate_grad_batches={base} is not divisible by {n_gpu} GPUs; "
                f"pass --accumulate_grad_batches explicitly")
        cfg.training.accumulate_grad_batches = base // n_gpu
    cfg.data.dataset_name = f"{cfg.data.dataset_name}_{save_name}"
    return cfg


def launch(cfg, train_set, val_set, test_set):
    """Single-GPU in-process, or NCCL DDP spawn across the requested GPUs."""
    gpu_ids = cfg.args.gpu_ids
    if gpu_ids.strip() == "-1":
        cfg.training.use_DDP = False
        cfg.training.device = [torch.device("cpu")]
        bbdm_main.set_random_seed(cfg.args.seed)
        runner = get_runner(cfg.runner, cfg)
        runner.train(train_set, val_set, test_set)
        return cfg

    gpu_list = [g for g in gpu_ids.split(",") if g != ""]
    if len(gpu_list) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        cfg.training.use_DDP = True
        torch.multiprocessing.spawn(
            _ddp_worker, args=(len(gpu_list), copy.deepcopy(cfg), train_set, val_set, test_set),
            nprocs=len(gpu_list), join=True)
    else:
        cfg.training.use_DDP = False
        cfg.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
        bbdm_main.set_random_seed(cfg.args.seed)
        runner = get_runner(cfg.runner, cfg)
        runner.train(train_set, val_set, test_set)
    return cfg


def _ddp_worker(rank, world_size, cfg, train_set, val_set, test_set):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = cfg.args.port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    bbdm_main.set_random_seed(cfg.args.seed)
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    cfg.training.device = [torch.device(f"cuda:{local_rank}")]
    cfg.training.local_rank = local_rank
    runner = get_runner(cfg.runner, cfg)
    runner.train(train_set, val_set, test_set)
    dist.destroy_process_group()


def legacy_eval(a, fold, save_name, eval_cfg, ckpt, first_gpu):
    """Original evaluation path: sample_to_eval_combined_with_uncertainty into samples_root."""
    eval_cfg.args.train = False
    eval_cfg.args.gpu_ids = first_gpu
    eval_cfg.training.use_DDP = False
    eval_cfg.training.device = [torch.device("cpu") if a.gpu_ids.strip() == "-1"
                                else torch.device(f"cuda:{first_gpu}")]
    eval_cfg.data.test.batch_size = 1
    eval_cfg.data.dataset_type = "custom_aligned"
    runner = get_runner(eval_cfg.runner, eval_cfg)
    net = runner.initialize_model(eval_cfg)
    print("loading", ckpt)
    net.load_state_dict(torch.load(ckpt, weights_only=True, map_location=eval_cfg.training.device[0])["model"])
    _, _, test_dataset = get_dataset(eval_cfg.data, fold["train"], fold["val"], fold["test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    sample_path = os.path.join(a.samples_root, save_name)
    runner.sample_to_eval_combined_with_uncertainty(net, test_loader, sample_path=sample_path)
    print(f"fold {fold['fold']} samples ->", sample_path, flush=True)


def find_latest_ckpt(results_root, dataset_name, model_name):
    ckpt_dir = os.path.join(results_root, dataset_name, model_name, "checkpoint")
    files = glob.glob(os.path.join(ckpt_dir, "top_model_epoch_*.pth"))
    if not files:
        files = glob.glob(os.path.join(ckpt_dir, "last_model.pth"))
    if not files:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    def epoch_of(f):
        m = re.search(r"epoch_(\d+)", f)
        return int(m.group(1)) if m else -1
    files.sort(key=epoch_of, reverse=True)
    return files[0]


def prune_checkpoints(ckpt_dir, keep):
    """Delete every .pth in ckpt_dir except `keep` (the checkpoint evaluation used).
    Optimizer states and the rotating last/latest copies are only needed to resume."""
    freed = 0
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth") and f != keep:
            fp = os.path.join(ckpt_dir, f)
            freed += os.path.getsize(fp)
            os.remove(fp)
    return freed


def main():
    a = parse_args()
    a.results_root = a.results_root or os.path.join(os.path.dirname(a.data_root.rstrip("/")), "results")
    a.samples_root = a.samples_root or os.path.join(os.path.dirname(a.results_root.rstrip("/")), "k-fold_samples")
    manifest = a.manifest or os.path.join(a.data_root, "manifest.csv")

    records = load_manifest(manifest, a.data_root)
    folds = make_folds(records, a.scheme, a.n_folds)
    rows = fold_table(folds, records)

    print(f"{len(records)} patches over {len({r['specimen'] for r in records})} specimens; "
          f"scheme={a.scheme}, {len(folds)} folds")
    print(format_fold_table(rows))
    for f in folds:
        assert_no_leakage(f, records)
    print(f"\nleakage assertion PASSED for all {len(folds)} folds "
          f"(no specimen appears in more than one partition)")

    os.makedirs(a.results_root, exist_ok=True)
    save_fold_assignments(folds, records, a.results_root)
    print("fold assignment table saved to", os.path.join(a.results_root, "fold_assignments.csv"))

    if a.dry_run:
        return

    wanted = None if a.folds is None else {int(x) for x in a.folds.split(",")}
    for fold in folds:
        if wanted is not None and fold["fold"] not in wanted:
            continue
        save_name = f"fold_{fold['fold']}_{a.tag}"
        print(f"\n=== fold {fold['fold']} ({save_name}) "
              f"test={fold['test_specimens']} val={fold['val_specimens']} ===", flush=True)
        cfg = build_config(a, save_name)

        if not a.skip_train:
            cfg = launch(cfg, fold["train"], fold["val"], fold["test"])

        # evaluation, always single GPU
        eval_cfg = build_config(a, save_name)
        ckpt = find_latest_ckpt(a.results_root, eval_cfg.data.dataset_name, eval_cfg.model.model_name)
        first_gpu = "0" if "CUDA_VISIBLE_DEVICES" in os.environ else a.gpu_ids.split(",")[0]
        if a.legacy_eval:
            legacy_eval(a, fold, save_name, eval_cfg, ckpt, first_gpu)
        else:
            deliv = a.deliverables_root or os.path.join(os.path.dirname(a.results_root.rstrip("/")), "deliverables")
            exp = a.experiment or a.tag
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_fold.py"),
                   "--config", a.config, "--ckpt", ckpt, "--data_root", a.data_root,
                   "--scheme", a.scheme, "--n_folds", str(a.n_folds), "--fold", str(fold["fold"]),
                   "--experiment", exp, "--out_root", deliv, "--gpu", first_gpu, "--seed", str(a.sample_seed),
                   "--n_gpus_train", str(max(1, len(a.gpu_ids.split(","))))]
            if a.vqgan_ckpt: cmd += ["--vqgan_ckpt", a.vqgan_ckpt]
            if a.manifest: cmd += ["--manifest", a.manifest]
            if a.eval_max_tiles: cmd += ["--max_tiles", str(a.eval_max_tiles)]
            print("eval:", " ".join(cmd), flush=True)
            import subprocess
            subprocess.run(cmd, check=True)
            print(f"fold {fold['fold']} deliverables ->", os.path.join(deliv, exp), flush=True)
        if not a.keep_optim:
            freed = prune_checkpoints(os.path.dirname(ckpt), keep=os.path.basename(ckpt))
            print(f"fold {fold['fold']}: pruned checkpoint dir to {os.path.basename(ckpt)} "
                  f"(+config.yaml), freed {freed/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
