import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import choice
from sklearn.preprocessing import StandardScaler
import torch
from biosppy.signals.tools import rms_error, pearson_correlation
import imageio
from joblib import Parallel, delayed
from pylab import rcParams
from similarity_measures import similaritymeasures

rcParams['figure.figsize'] = 17, 15

def noisy_labels(y, p_flip):
    n_select = int(p_flip * y.shape[2])
    flip_ix = choice([i for i in range(y.shape[2])], size=n_select)
    y[:, :, flip_ix] = 1 - y[:, :, flip_ix]


def sample_images(dataset_name, batches_done, val_dataloader, generator, from_ppg, from_latent_ecg, device):
    """Saves a generated sample from the validation set"""

    generator.eval()

    signals = next(iter(val_dataloader))
    real_A = signals[0].to(device)
    real_B = signals[1].to(device)
    fake_B = generator(real_A)

    real_A = torch.squeeze(real_A).cpu().detach().numpy().T
    real_B = torch.squeeze(real_B).cpu().detach().numpy().T
    fake_B = torch.squeeze(fake_B).cpu().detach().numpy().T

    if from_latent_ecg:
        real_A0 = signals[2].to(device)
        real_A0 = torch.squeeze(real_A0).cpu().detach().numpy().T
        num_charts = 4
        plot_components = [real_A0, real_A, real_B, fake_B]
    else:
        num_charts = 3
        plot_components = [real_A, real_B, fake_B]

    img_dir = "sample_signals/%s/%s.png" % (dataset_name, batches_done)
    fig = plt.figure()

    for i in range(num_charts):
        plt.subplot(num_charts, 1, i + 1)
        if from_latent_ecg:
            if i == 0:
                plt.title('Real ECG')
            elif i == 1:
                plt.title('Latent ECG')
            elif i == 2:
                plt.title('Real PPG')
            else:
                plt.title('Fake PPG')

        elif not from_ppg:
            if i == 0:
                plt.title('Real ECG')
            elif i == 1:
                plt.title('Real PPG')
            else:
                plt.title('Fake PPG')

        else:
            if i == 0:
                plt.title('Real PPG')
            elif i == 1:
                plt.title('Real ECG')
            else:
                plt.title('Fake ECG')

        plt.plot(plot_components[i])
        plt.axis('off')

    # plt.tight_layout()
    plt.savefig(img_dir)
    plt.close(fig)


def sample_images2(dataset_name, val_dataloader, generator, images, writer, steps, device):
    """Saves a generated sample from the validation set"""

    generator.eval()

    current_img_dir = "sample_signals/%s/%s.png" % (dataset_name, steps)
    gif_img_dir = "sample_signals/%s/progress.gif" % (dataset_name)

    signals = next(iter(val_dataloader))
    real_A = signals[0].to(device)
    real_B = signals[1].to(device)
    fake_B = generator(real_A)

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
    # ncols, nrows = fig.canvas.get_width_height()
    # image_from_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(nrows, ncols, 3)[:, :, 2:3]
    fig.savefig(current_img_dir)
    # writer.add_image('images', image_from_plot, steps, dataformats='HWC')
    # images.append(image_from_plot)
    # imageio.mimsave(gif_img_dir, images)
    plt.close(fig)

def eval_rmse_p(signal_a, signal_b):
    #signal_b, _ = smoother(signal_b, size=2)

    rmse = rms_error(signal_a, signal_b)['rmse']
    p = scipy.stats.pearsonr(signal_a, signal_b)[0]

    return rmse, p


def evaluate_generated_signal_quality(val_dataloader, generator, writer, steps, device):
    generator.eval()

    all_signal = []
    all_generated_signal = []
    all_signal_ = []
    all_generated_signal_ = []

    for _, batch in enumerate(val_dataloader):
        real_A = batch[0].to(device)

        real_B = batch[1].to(device)
        real_B = torch.squeeze(real_B).cpu().detach().numpy()
        #print(real_A.shape)
        fake_B = generator(real_A)
        fake_B = torch.squeeze(fake_B).cpu().detach().numpy()

        all_signal.append(real_B.flatten())
        all_generated_signal.append(fake_B.flatten())

        all_signal_.append(real_B)
        all_generated_signal_.append(fake_B)

    whole_signal = np.concatenate(all_signal)
    whole_generated_signal = np.concatenate(all_generated_signal)

    #scaler = StandardScaler()
    #whole_signal_ = scaler.fit_transform(np.reshape(whole_signal, (-1, 1)))
    #whole_generated_signal_ = scaler.fit_transform(np.reshape(whole_generated_signal, (-1, 1)))

    #sns_plot = sns.distplot(whole_signal_)
    #figure = sns_plot.get_figure()
    #figure.savefig("ecg_label_dist.png", dpi=400)

    #sns_plot = sns.distplot(whole_generated_signal_)
    #figure = sns_plot.get_figure()
    #figure.savefig("ecg_gen_dist.png", dpi=400)


    rmse = rms_error(whole_signal, whole_generated_signal)
    p1 = pearson_correlation(whole_signal, whole_generated_signal)['rxy']
    #p2 = scipy.stats.pearsonr(whole_signal, whole_generated_signal)

    all_signal = np.vstack(all_signal_)
    all_generated_signal = np.vstack(all_generated_signal_)

    rmse_p_pairs = Parallel(n_jobs=8)(delayed(eval_rmse_p)(signal_a, signal_b) for signal_a, signal_b in zip(all_signal, all_generated_signal))
    res = list(zip(*rmse_p_pairs))

    rmse_mean, rmse_std = np.mean(res[0]), np.std(res[0])
    p_mean, p_std = np.mean(res[1]), np.std(res[1])

    print('\nepoch: ', steps)
    print('rms_error: ', rmse)
    print('pearson_correlation_biosppy: ', p1)
    #print('pearson_correlation_scipy: ', p2[0], p2[1])
    print(rmse_mean, rmse_std)
    print(p_mean, p_std)


    fdists = []
    fdists = Parallel(n_jobs=32)(delayed(similaritymeasures.frechet_dist)(sig, all_signal[i]) for i, sig in enumerate(all_generated_signal))
    print('frechet distance: ', np.mean(fdists), np.std(fdists))

    if writer:
        writer.add_scalars('losses4', {'rms_error': rmse['rmse']}, steps)
        writer.add_scalars('losses4', {'pearson_correlation': p1}, steps)
        writer.add_scalars('losses4', {'p_mean': p_mean}, steps)
        


def generate_ecg(generated_fi, dataloader, generator, device):
    generator.eval()

    all_generated_signal = []
    for _, batch in enumerate(dataloader):
        real_A = batch[0].to(device)
        fake_B = generator(real_A)
        fake_B = torch.squeeze(fake_B).cpu().detach().numpy()
        all_generated_signal.append(fake_B)

    with open(generated_fi, 'wb') as f:
        np.save(f, all_generated_signal)
        print(generated_fi, np.array(all_generated_signal).shape)
