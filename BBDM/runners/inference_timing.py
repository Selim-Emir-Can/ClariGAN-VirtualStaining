"""Inference-timing instrumentation (report-only).

Added to answer the reviewer request for inference time per 256x256 patch,
whole-field inference time, and GPU model / memory.

This module is purely observational.  It wraps a timer around the existing
``net.sample()`` calls and prints a summary; it does not touch tensors,
control flow, RNG state, or anything written to disk by the pipeline.  Every
entry point is wrapped in try/except so an instrumentation failure can never
take down a run.

To remove the instrumentation entirely, delete this file and the three blocks
marked ``--- inference timing (report-only)`` in
``runners/DiffusionBasedModelRunners/BBDMRunner.py`` and
``BBDMRunner_pixel_loss.py``.
"""

import json
import os
import re
import statistics
import time
from contextlib import contextmanager

import torch

# Patch names look like:  R1-G_row3_col7_10x10  /  R3-A_row0_col2_5x5
_FIELD_RE = re.compile(r'^(?P<field>.+?)_row\d+_col\d+_(?P<grid>\d+x\d+)')

_GIB = 1024 ** 3


def _fmt_s(seconds):
    """Seconds -> compact human string, without losing precision for small values."""
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{seconds:.1f} s ({int(m)} min {s:04.1f} s)"
    h, m = divmod(m, 60)
    return f"{seconds:.0f} s ({int(h)} h {int(m)} min)"


def _scan_field_sizes(dataset_path):
    """Count how many 256x256 patches actually belong to each imaging field.

    Reads only filenames from ``<dataset_path>/train/A``.  Returns
    ``{grid_label: [patch_count_per_field, ...]}``.  The dataset is
    tissue-filtered, so these are the real per-field patch counts rather than
    the nominal grid implied by the ``5x5`` / ``10x10`` label.
    """
    counts = {}
    src = os.path.join(dataset_path, 'train', 'A')
    if not os.path.isdir(src):
        return counts
    per_field = {}
    for fname in os.listdir(src):
        m = _FIELD_RE.match(os.path.splitext(fname)[0])
        if not m:
            continue
        key = (m.group('grid'), m.group('field'))
        per_field[key] = per_field.get(key, 0) + 1
    for (grid, _field), n in per_field.items():
        counts.setdefault(grid, []).append(n)
    for grid in counts:
        counts[grid].sort()
    return counts


class InferenceTimer:
    """Accumulates wall-clock around each ``net.sample()`` call."""

    def __init__(self, config, sample_path=None, tag=''):
        self.enabled = True
        self.tag = tag
        self.sample_path = sample_path
        self.durations = []          # one entry per net.sample() call
        self.warmup = None           # first call, reported separately
        self.patch_names = []
        self.observed_batches = set()  # patches per net.sample() call, as actually run
        self.dataset_path = None
        self.sample_step = None
        self.sample_num = 1
        self.batch_size = 1
        self.image_size = None
        self.device_index = 0
        try:
            self.sample_step = config.model.BB.params.sample_step
            self.sample_num = config.testing.sample_num
            self.batch_size = config.data.test.batch_size
            self.image_size = config.data.dataset_config.image_size
            self.dataset_path = config.data.dataset_config.dataset_path
            dev = config.training.device[0]
            self.device_index = dev.index if getattr(dev, 'index', None) is not None else 0
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device_index)
        except Exception:
            pass

    @contextmanager
    def record(self, name=None, batch=1):
        """Time the enclosed sampling call.  Yields nothing, changes nothing.

        ``batch`` is the number of patches passed to this single ``net.sample()``
        call, which is what the reported per-patch figure is normalised by.
        """
        if not self.enabled:
            yield
            return
        try:
            self.observed_batches.add(int(batch))
        except Exception:
            pass
        cuda = torch.cuda.is_available()
        try:
            if cuda:
                torch.cuda.synchronize(self.device_index)
        except Exception:
            pass
        t0 = time.perf_counter()
        try:
            yield
        finally:
            try:
                if cuda:
                    torch.cuda.synchronize(self.device_index)
                dt = time.perf_counter() - t0
                if self.warmup is None:
                    self.warmup = dt          # first call carries CUDA/cuDNN warm-up
                else:
                    self.durations.append(dt)
                    if name is not None:
                        self.patch_names.append(str(name))
            except Exception:
                pass

    # ---------------------------------------------------------------- report
    def _stats(self):
        d = self.durations
        if not d:
            return None
        return {
            'n_calls': len(d),
            'mean': statistics.mean(d),
            'median': statistics.median(d),
            'sd': statistics.pstdev(d) if len(d) > 1 else 0.0,
            'min': min(d),
            'max': max(d),
        }

    def report(self):
        try:
            return self._report()
        except Exception as exc:                      # never break a run
            print(f'[inference-timing] skipped: {exc}')
            return None

    def _report(self):
        st = self._stats()
        if st is None:
            return None

        gpu_name, gpu_total = 'cpu / unknown', None
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.device_index)
                gpu_name = props.name
                gpu_total = props.total_memory / _GIB
        except Exception:
            pass

        peak_alloc = peak_res = None
        try:
            if torch.cuda.is_available():
                peak_alloc = torch.cuda.max_memory_allocated(self.device_index) / _GIB
                peak_res = torch.cuda.max_memory_reserved(self.device_index) / _GIB
        except Exception:
            pass

        # Patches per net.sample() call, as actually executed.  The uncertainty
        # path samples one patch at a time regardless of config test batch size.
        call_batch = max(self.observed_batches) if self.observed_batches else 1
        per_patch_1 = st['median'] / call_batch        # one diffusion draw
        per_patch_n = per_patch_1 * self.sample_num    # the pipeline as actually run

        fields = _scan_field_sizes(self.dataset_path) if self.dataset_path else {}

        w = 74
        lines = []
        lines.append('=' * w)
        title = 'Inference timing (report-only)'
        if self.tag:
            title += f' - {self.tag}'
        lines.append(title)
        lines.append('=' * w)
        lines.append(f'{"GPU":<32}: {gpu_name}'
                     + (f' ({gpu_total:.2f} GiB total)' if gpu_total else ''))
        lines.append(f'{"torch / CUDA":<32}: {torch.__version__} / '
                     f'{torch.version.cuda or "n-a"}')
        lines.append(f'{"Diffusion steps (sample_step)":<32}: {self.sample_step}')
        lines.append(f'{"Draws per patch (sample_num)":<32}: {self.sample_num}')
        lines.append(f'{"Patch size":<32}: {self.image_size}x{self.image_size}')
        lines.append(f'{"Patches per sample() call":<32}: {call_batch}'
                     f'  (dataloader test batch: {self.batch_size})')
        lines.append('-' * w)
        lines.append(f'{"net.sample() calls timed":<32}: {st["n_calls"]}'
                     + (f'  (warm-up call {self.warmup:.3f} s excluded)'
                        if self.warmup is not None else ''))
        lines.append(f'{f"Per {self.image_size}x{self.image_size} patch, 1 draw":<32}: '
                     f'median {per_patch_1:.3f} s | '
                     f'mean {st["mean"] / call_batch:.3f} s | '
                     f'sd {st["sd"] / call_batch:.3f} s')
        lines.append(f'{"":<32}  min {st["min"] / call_batch:.3f} s | '
                     f'max {st["max"] / call_batch:.3f} s')
        lines.append(f'{f"Per patch, {self.sample_num} draws (as run)":<32}: '
                     f'{_fmt_s(per_patch_n)}')
        if peak_alloc is not None:
            lines.append(f'{"Peak GPU memory":<32}: {peak_alloc:.2f} GiB allocated / '
                         f'{peak_res:.2f} GiB reserved')

        if fields:
            lines.append('-' * w)
            lines.append('Whole-field inference (patches per field measured from the')
            lines.append('tissue-filtered dataset, not the nominal grid label):')
            for grid in sorted(fields):
                counts = fields[grid]
                n_fields = len(counts)
                mean_p = statistics.mean(counts)
                max_p = max(counts)
                lines.append(f'  {grid:<7} n={n_fields:<3} patches/field '
                             f'mean {mean_p:.1f}, max {max_p}')
                lines.append(f'  {"":<7} 1 draw : mean {_fmt_s(mean_p * per_patch_1)}'
                             f' | worst {_fmt_s(max_p * per_patch_1)}')
                lines.append(f'  {"":<7} {self.sample_num} draws: '
                             f'mean {_fmt_s(mean_p * per_patch_n)}'
                             f' | worst {_fmt_s(max_p * per_patch_n)}')
        lines.append('=' * w)

        text = '\n'.join(lines)
        print(text)

        payload = {
            'tag': self.tag,
            'gpu': gpu_name,
            'gpu_total_gib': gpu_total,
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'sample_step': self.sample_step,
            'sample_num': self.sample_num,
            'patches_per_sample_call': call_batch,
            'dataloader_test_batch_size': self.batch_size,
            'patch_size': self.image_size,
            'n_calls_timed': st['n_calls'],
            'warmup_call_s': self.warmup,
            'per_patch_1_draw_s': {k: st[k] / call_batch for k in
                                   ('mean', 'median', 'sd', 'min', 'max')},
            'per_patch_all_draws_s': per_patch_n,
            'peak_gpu_alloc_gib': peak_alloc,
            'peak_gpu_reserved_gib': peak_res,
            'patches_per_field': fields,
            'all_call_durations_s': self.durations,
        }
        if self.sample_path:
            try:
                os.makedirs(self.sample_path, exist_ok=True)
                out = os.path.join(self.sample_path, 'inference_timing.json')
                with open(out, 'w') as fh:
                    json.dump(payload, fh, indent=2)
                print(f'[inference-timing] written to {out}')
            except Exception as exc:
                print(f'[inference-timing] could not write json: {exc}')
        return payload
