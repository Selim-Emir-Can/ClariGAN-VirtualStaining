"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import os
import copy
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def run_val(model, val_dataset):
    """Forward-only pass over val. Returns mean per-element L1 between fake_B
    and real_B. Does not affect training state on exit."""
    model.eval()
    val_l1_sum = 0.0
    val_n = 0
    with torch.no_grad():
        for vd in val_dataset:
            model.set_input(vd)
            model.test()
            bsz = model.fake_B.size(0)
            val_l1_sum += F.l1_loss(model.fake_B, model.real_B).item() * bsz
            val_n += bsz
    # Restore train mode -- BaseModel has eval() but no symmetric train()
    for name in model.model_names:
        if isinstance(name, str):
            getattr(model, "net" + name).train()
    return val_l1_sum / max(val_n, 1)


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Optional validation monitoring (NO checkpoint selection -- just logged).
    val_dataset = None
    val_log_path = None
    if getattr(opt, "val_dataroot", ""):
        val_opt = copy.deepcopy(opt)
        val_opt.dataroot = opt.val_dataroot
        val_opt.isTrain = False
        val_opt.phase = "val"
        val_opt.serial_batches = True
        val_opt.no_flip = True
        val_opt.batch_size = 1
        val_dataset = create_dataset(val_opt)
        val_log_path = os.path.join(opt.checkpoints_dir, opt.name, "val_log.txt")
        os.makedirs(os.path.dirname(val_log_path), exist_ok=True)
        print(f"validation monitoring enabled: {len(val_dataset)} val pairs, log -> {val_log_path}")

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        total_epochs = opt.niter + opt.niter_decay
        pbar = tqdm(dataset, total=len(dataset), desc=f"epoch {epoch}/{total_epochs}", dynamic_ncols=True, leave=True)
        for i, data in enumerate(pbar):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(opt.critic_iters)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                pbar.set_postfix({k: f"{v:.3f}" for k, v in losses.items()})

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                tqdm.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        pbar.close()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        if val_dataset is not None:
            val_l1 = run_val(model, val_dataset)
            msg = f"[VAL] epoch {epoch} L1 = {val_l1:.6f}"
            print(msg)
            with open(val_log_path, "a") as f:
                f.write(msg + "\n")
