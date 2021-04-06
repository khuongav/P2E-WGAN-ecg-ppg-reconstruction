from joblib import Parallel, delayed
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = 17, 15

# import similaritymeasures


def compute_gradient_penalty(D, real_samples, fake_samples, real_A, patch, device):
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


mean_conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4,
                            bias=False, padding_mode='replicate', padding=1)
mean_conv.weight.data = torch.full_like(
    mean_conv.weight.data, 0.25)
pad = torch.nn.ReplicationPad1d((0, 1))


def smoother(fake, device):
    fake = mean_conv.to(device)(fake)
    fake = pad.to(device)(fake)
    return fake


def sample_images(dataset_name, val_dataloader, generator, steps, device):
    """Saves generated signals from the validation set"""

    generator.eval()

    current_img_dir = "sample_signals/%s/%s.png" % (dataset_name, steps)

    signals = next(iter(val_dataloader))
    real_A = signals[0].to(device)
    real_B = signals[1].to(device)
    fake_B = generator(real_A)
    fake_B = smoother(fake_B, device)

    real_A = torch.squeeze(real_A).cpu().detach().numpy()
    real_B = torch.squeeze(real_B).cpu().detach().numpy()
    fake_B = torch.squeeze(fake_B).cpu().detach().numpy()

    fig, axes = plt.subplots(real_A.shape[0], 3)

    axes[0][0].set_title('Real A')
    axes[0][1].set_title('Real B')
    axes[0][2].set_title('Fake B')

    for idx, signal in enumerate(real_A):
        axes[idx][0].plot(real_A[idx], color='c')
        axes[idx][1].plot(real_B[idx], color='m')
        axes[idx][2].plot(fake_B[idx], color='y')

    fig.canvas.draw()
    fig.savefig(current_img_dir)
    plt.close(fig)


def eval_rmse_p(signal_a, signal_b):
    rmse = np.sqrt(((signal_a - signal_b) ** 2).mean())
    p = stats.pearsonr(signal_a, signal_b)[0]

    return rmse, p


def evaluate_generated_signal_quality(val_dataloader, generator, writer, steps, device):
    generator.eval()

    all_signal_ = []
    all_generated_signal_ = []

    for _, batch in enumerate(val_dataloader):
        real_A = batch[0].to(device)

        real_B = batch[1].to(device)
        real_B = torch.squeeze(real_B).cpu().detach().numpy()

        fake_B = generator(real_A)
        fake_B = smoother(fake_B, device)
        fake_B = torch.squeeze(fake_B).cpu().detach().numpy()

        all_signal_.append(real_B)
        all_generated_signal_.append(fake_B)

    all_signal = np.vstack(all_signal_)
    all_generated_signal = np.vstack(all_generated_signal_)

    rmse_p_pairs = Parallel(n_jobs=8)(delayed(eval_rmse_p)(
        signal_a, signal_b) for signal_a, signal_b in zip(all_signal, all_generated_signal))
    res = list(zip(*rmse_p_pairs))

    rmse_mean, rmse_std = np.mean(res[0]), np.std(res[0])
    p_mean, p_std = np.mean(res[1]), np.std(res[1])

    print('\nepoch: ', steps)
    print('rmse_mean:', rmse_mean, ', rmse_std:', rmse_std)
    print('p_mean:', p_mean, ', p_std:', p_std)
    # fdists = []
    # fdists = Parallel(n_jobs=32)(delayed(similaritymeasures.frechet_dist)(sig, all_signal[i]) for i, sig in enumerate(all_generated_signal))
    # print('frechet distance: ', np.mean(fdists), np.std(fdists))

    if writer:
        writer.add_scalars('losses4', {'rms_error': rmse_mean}, steps)
        writer.add_scalars('losses4', {'p_mean': p_mean}, steps)
