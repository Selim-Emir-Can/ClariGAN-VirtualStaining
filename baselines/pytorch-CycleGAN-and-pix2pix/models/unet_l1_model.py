"""Deterministic U-Net regression baseline: the pix2pix generator trained with L1 only.

No discriminator, no adversarial term, no perceptual term. Same generator capacity as the
pix2pix baseline (netG unet_256, ngf as given). At inference use --eval: dropout off and
batch-norm statistics fixed, so one deterministic output per tile.

Requested by reviewer (CRITICAL 10): "a deterministic U-Net with L1".
"""
import torch
from .base_model import BaseModel
from . import networks


class UnetL1Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.add_argument("--lambda_L1", type=float, default=1.0, help="weight for L1 loss (the only loss)")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_L1"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G"]
        self.device = opt.device
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain)
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_L1.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
