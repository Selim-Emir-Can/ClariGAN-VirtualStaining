"""Standalone inference-time measurement for the reviewer's Methods numbers.

Reports inference time per 256x256 patch, whole-field inference time, and the
GPU model / memory, using the same `InferenceTimer` that instruments the
k-fold scripts -- so the printed block is identical in format to what a real
`k-fold_validation*.py` run produces.

Why this exists: the k-fold test pass only runs after a fold has trained
(~4.4 h/fold), and no ClariGAN fold checkpoint is currently on disk.  Diffusion
sampling cost is fixed by the architecture and `sample_step`, not by the values
of the weights, so this gives the identical timing without retraining.  Pass
`--ckpt` once a fold checkpoint exists to confirm on the real weights.

Usage
-----
    python measure_inference_time.py
    python measure_inference_time.py --n 30 --warmup 3
    python measure_inference_time.py --ckpt results/<fold>/LBBDM-f16/checkpoint/top_model_epoch_40.pth
    python measure_inference_time.py --config configs/Template-BBDM_pixel_256_matchUNet.yaml
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import dict2namespace                       # noqa: E402
from runners.inference_timing import InferenceTimer    # noqa: E402

DEFAULT_CONFIG = os.path.join(
    'configs', 'Template-LBBDM-f16_imagenetVQGAN_finetuned.yaml')


def load_config(path):
    with open(path, 'r') as fh:
        raw = yaml.load(fh, Loader=yaml.UnsafeLoader)
    return raw if hasattr(raw, 'model') else dict2namespace(raw)


def build_net(config, device):
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
    from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
    if config.model.model_type == 'LBBDM':
        return LatentBrownianBridgeModel(config.model).to(device)
    return BrownianBridgeModel(config.model).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--ckpt', default=None,
                    help='optional trained checkpoint; timing is weight-independent')
    ap.add_argument('--n', type=int, default=25, help='timed sample() calls')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    config.training.device = [device]

    net = build_net(config, device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location='cpu')
        net.load_state_dict(state['model'] if 'model' in state else state, strict=False)
        print(f'[weights] loaded {args.ckpt}')
    else:
        print('[weights] randomly initialised -- sampling cost is independent of '
              'weight values, so per-patch timing is unaffected.')
    net.eval()

    size = config.data.dataset_config.image_size
    ch = config.data.dataset_config.channels
    timer = InferenceTimer(config, sample_path=None,
                           tag=f'{config.model.model_name} (standalone)')

    x_cond = torch.randn(1, ch, size, size, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            net.sample(x_cond, clip_denoised=config.testing.clip_denoised)
        for _ in range(args.n + 1):        # +1: timer discards its first call
            with timer.record(name='benchmark', batch=x_cond.shape[0]):
                net.sample(x_cond, clip_denoised=config.testing.clip_denoised)

    timer.report()


if __name__ == '__main__':
    main()
