import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    # ---------------- reference / context conditioning ----------------
    # BB.params.ref_cond = {source: target|input, k: int}. The bank holds, for every TRAINING tile,
    # the channel mean and std of its frozen-VQGAN latent (256 + 256 = 512 dims) computed from its
    # stained target ("target", variant A) or its uncleared input ("input", variant B). A sample's
    # reference vector is the mean feature of other training tiles of the same specimen, never from
    # its own crop unit (a 5x5 crop and the 10x10 crops inside it), so a training tile cannot see
    # its own ground truth. Training: k random references; eval: all of them (deterministic).
    @torch.no_grad()
    def build_ref_bank(self, pairs, split_file=None, manifest=None, batch=32):
        import csv, os
        from PIL import Image
        import numpy as np
        from specimen_labels import SPECIMENS, specimen_of
        rc = self.model_config.BB.params.ref_cond
        source, dev = rc.source, next(self.parameters()).device
        unit_of = {}
        if split_file:
            unit = {r["tile_id"]: r["unit"] for r in csv.DictReader(open(split_file))}
            man = manifest or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(pairs[0][0]))), "manifest.csv")
            for r in csv.DictReader(open(man)):
                u = unit.get(r["tile_id"])
                for k in (r["tile_id"], os.path.splitext(r["input_filename"])[0], os.path.splitext(r["target_filename"])[0]):
                    unit_of[k] = u
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        paths = [t if source == "target" else i for i, t in pairs]
        feats = []
        for b0 in range(0, len(paths), batch):
            xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
                  for p in paths[b0:b0 + batch]]
            z = self.encode(torch.stack(xs).to(dev), cond=(source == "input"))
            feats.append(torch.cat([z.mean((2, 3)), z.std((2, 3))], 1).float())
        self._ref_feats = torch.cat(feats)                                              # [N, 512]
        self._ref_spec = torch.tensor([SPECIMENS.index(specimen_of(stem(i))) for i, _ in pairs], device=dev)
        self._ref_unit = [unit_of.get(stem(i), stem(i)) for i, _ in pairs]
        self._unit_of = unit_of
        print(f"reference bank ({source}): {len(pairs)} training tiles, feature {tuple(self._ref_feats.shape)}", flush=True)

    def ref_features(self, names):
        if getattr(self, "_ref_feats", None) is None:
            raise RuntimeError("ref_cond is set but build_ref_bank() was not called")
        import os
        from specimen_labels import SPECIMENS, specimen_of
        k = int(self.model_config.BB.params.ref_cond.k)
        out = []
        for n in names:
            st = os.path.splitext(os.path.basename(str(n)))[0]
            u = self._unit_of.get(st, st)
            idx = [i for i in (self._ref_spec == SPECIMENS.index(specimen_of(st))).nonzero().flatten().tolist()
                   if self._ref_unit[i] != u]
            if self.training and len(idx) > k:
                idx = [idx[j] for j in torch.randperm(len(idx))[:k].tolist()]
            out.append(self._ref_feats[idx].mean(0))
        return torch.stack(out)

    def _ref(self, ref_names):
        if ref_names is None or not self.model_config.BB.params.__contains__("ref_cond"):
            return None
        return self.ref_features(ref_names)

    def forward(self, x, x_cond, context=None, class_y=None, ref_names=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        context = self.get_cond_stage_context(x_cond)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context, class_y=class_y,
                               ref=self._ref(ref_names))

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent
    
    # def encode_mulitmodal(self, x, cond=True, normalize=None):
    #     normalize = self.model_config.normalize_latent if normalize is None else normalize
    #     model = self.vqgan

    #     xp_latent = torch.zeros((x.shape[0], 2*256, 16, 16), dtype=x.dtype, device=x.device)

    #     for i in range(2):
    #         start = i*256
    #         end = (i+1)*256
    #         x_latent = model.encoder(x[:,i,:,:,:])

    #         if not self.model_config.latent_before_quant_conv:
    #             x_latent = model.quant_conv(x_latent)
    #         if normalize:
    #             if cond:
    #                 x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
    #             else:
    #                 x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
            
    #         xp_latent[:,start:end,:,:] = x_latent
    #     return xp_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    # def decode_multimodal(self, x_latent, cond=True, normalize=None):
    #     normalize = self.model_config.normalize_latent if normalize is None else normalize
    #     if normalize:
    #         if cond:
    #             x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
    #         else:
    #             x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
    #     model = self.vqgan
    #     x_latent = x_latent[:,int(1*256):,:,:]
    #     if self.model_config.latent_before_quant_conv:
    #         x_latent = model.quant_conv(x_latent)
    #     x_latent_quant, loss, _ = model.quantize(x_latent)
    #     out = model.decode(x_latent_quant)
    #     return out
    
    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False, class_y=None, ref_names=None):
        x_cond_latent = self.encode(x_cond, cond=True)
        ref = self._ref(ref_names)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step,
                                      class_y=class_y, ref=ref)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step,
                                      class_y=class_y, ref=ref)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
