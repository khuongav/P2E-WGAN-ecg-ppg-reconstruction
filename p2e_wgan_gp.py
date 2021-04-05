import torch.autograd as autograd
import argparse
import time
import datetime
import os
import sys
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from models import weights_init_normal, GeneratorUNet, Discriminator
from data import get_data_loader
from utils import sample_images, evaluate_generated_signal_quality

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--shuffle_training", type=bool,
                    default=True, help="shuffle training")
parser.add_argument("--is_eval", type=bool,
                    default=False, help="evaluation mode")
parser.add_argument("--from_ppg", type=bool, default=True,
                    help="reconstruct from ppg")
parser.add_argument("--dataset_name", type=str,
                    default="mimic_mse_peaks_only_cleaned_wgan_gp", help="name of the dataset")
parser.add_argument("--n_epochs", type=int, default=10000,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=192,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--signal_length", type=int,
                    default=375, help="size of the signal")
parser.add_argument("--checkpoint_interval", type=int,
                    default=30, help="interval between model checkpoints")

args, unknown = parser.parse_known_args()
print(args)


gen_weight = 1
disc_weight = 1
lambda_pixel = 20
rpeak_weight = 4
#opeak_weight = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

dataloader, val_dataloader = get_data_loader(args.batch_size, from_ppg=args.from_ppg,
                                             shuffle_training=args.shuffle_training)

# Loss functions
criterion_pixelwise = torch.nn.MSELoss(reduction='sum')

# Calculate output of signal discriminator (PatchGAN)
patch = (1, 9)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    if torch.cuda.device_count() > 1:
        # if False:
        generator = torch.nn.DataParallel(
            generator, device_ids=[1, 2, 3]).to(device)
        discriminator = torch.nn.DataParallel(
            discriminator, device_ids=[1, 2, 3]).to(device)
    else:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    criterion_pixelwise.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))


def compute_gradient_penalty(D, real_samples, fake_samples, real_A):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha)
                                            * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, real_A)
    fake = torch.full(
        (real_samples.shape[0], *patch), 1, dtype=torch.float, device=device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if args.epoch != 0:
    # Load pretrained models

    pretrained_path = "saved_models/%s/multi_models_%d.pth" % (
        args.dataset_name, args.epoch)
    checkpoint = torch.load(pretrained_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    if args.is_eval:
        evaluate_generated_signal_quality(
            val_dataloader, generator, None, args.epoch, device)
        sys.exit()
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

os.makedirs("saved_models/%s" % args.dataset_name, exist_ok=True)
os.makedirs("sample_signals/%s" % args.dataset_name, exist_ok=True)
os.makedirs("logs/%s" % args.dataset_name, exist_ok=True)
writer = SummaryWriter("logs/%s" % args.dataset_name)
images = []

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(args.epoch+1, args.n_epochs):

    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = batch[0].to(device)
        real_B = batch[1].to(device)

        if i % 3 == 0:

            # ------------------
            #  Train Generators
            # ------------------
            generator.train()
            for p in generator.parameters():
                p.grad = None

            # GAN loss
            fake_B = generator(real_A)

            # Pixel-wise loss
            if args.from_ppg:
                opeaks = batch[2].to(device)
                rpeaks = batch[3].to(device)
                fake_B_masked_opeaks = fake_B * (opeaks != 0)
                fake_B_masked_rpeaks = fake_B * (rpeaks != 0)
                opeak_count = torch.sum(opeaks != 0)
                rpeak_count = torch.sum(rpeaks != 0)

                loss_pixel_opeaks = criterion_pixelwise(
                    fake_B_masked_opeaks, opeaks)
                loss_pixel_rpeaks = criterion_pixelwise(
                    fake_B_masked_rpeaks, rpeaks)
                loss_pixel = loss_pixel_opeaks / opeak_count + \
                    rpeak_weight * loss_pixel_rpeaks / rpeak_count
                #loss_pixel = loss_pixel + criterion_pixelwise(fake_B, real_B)
            else:
                loss_pixel = criterion_pixelwise(fake_B, real_B)

            mean_conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4,
                                        bias=False, padding_mode='replicate', padding=1).to(device)
            mean_conv.weight.data = torch.full_like(
                mean_conv.weight.data, 0.25).to(device)
            fake_B = mean_conv(fake_B)
            pad = torch.nn.ReplicationPad1d((0, 1))
            fake_B = pad(fake_B)

            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = -torch.mean(pred_fake)

            # Total loss
            loss_G = gen_weight * loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

        else:
            fake_B = generator(real_A)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # optimizer_D.zero_grad()
        for p in discriminator.parameters():
            p.grad = None
        # Real images
        real_validity = discriminator(real_B, real_A)
        # Fake images
        fake_validity = discriminator(fake_B.detach(), real_A)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_B, fake_B.detach(), real_A)
        # Adversarial loss
        loss_D0 = -torch.mean(real_validity) + \
            torch.mean(fake_validity) + 10 * gradient_penalty
        loss_D = loss_D0 * disc_weight

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D0 loss: %f] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (epoch, args.n_epochs, i, len(dataloader),
               loss_D0.item(), loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item(),
               time_left)
        )

        writer.add_scalars('losses', {'g_loss': loss_G.item()}, batches_done)
        writer.add_scalars('losses', {'d_loss': loss_D.item()}, batches_done)
        writer.add_scalars(
            'losses2', {'d_loss0': loss_D0.item()}, batches_done)
        writer.add_scalars(
            'losses2', {'gan_loss': loss_GAN.item()}, batches_done)
        writer.add_scalars(
            'losses3', {'pixel_loss': loss_pixel.item()}, batches_done)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, "saved_models/%s/multi_models_%d.pth" % (args.dataset_name, epoch))
        sample_images(args.dataset_name, val_dataloader,
                      generator, images, writer, epoch, device)
        evaluate_generated_signal_quality(
            val_dataloader, generator, writer, epoch, device)
