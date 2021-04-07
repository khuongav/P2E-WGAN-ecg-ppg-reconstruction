import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BioData(Dataset):
    def __init__(self, from_data_path, to_data_path, transformer, ecg_peaks_data_path):

        with open(from_data_path, 'rb') as f:
            self.from_dataset = np.load(f)
            print(from_data_path, self.from_dataset.shape)

        with open(to_data_path, 'rb') as f:
            self.to_dataset = np.load(f)
            print(to_data_path, self.to_dataset.shape)

        if ecg_peaks_data_path:
            self.from_ppg = True
            with open(ecg_peaks_data_path['opeaks'], 'rb') as f:
                self.opeaks_dataset = np.load(f)
                print(ecg_peaks_data_path['opeaks'], self.opeaks_dataset.shape)

            with open(ecg_peaks_data_path['rpeaks'], 'rb') as f:
                self.rpeaks_dataset = np.load(f)
                print(ecg_peaks_data_path['rpeaks'], self.rpeaks_dataset.shape)
        else:
            self.from_ppg = False

        self.transformer = transformer

    def __len__(self):
        return self.from_dataset.shape[0]

    def __getitem__(self, idx):
        X = self.from_dataset[idx][np.newaxis, :]
        y = self.to_dataset[idx][np.newaxis, :]
        X = self.transformer(X)
        y = self.transformer(y)

        if self.from_ppg:
            opeaks = self.opeaks_dataset[idx][np.newaxis, :]
            rpeaks = self.rpeaks_dataset[idx][np.newaxis, :]
            opeaks = self.transformer(opeaks)
            rpeaks = self.transformer(rpeaks)

        if self.from_ppg:
            return X.float(), y.float(), opeaks.float(), rpeaks.float()
        else:
            return X.float(), y.float()


class NP_to_Tensor(object):
    def __call__(self, sample):
        return torch.tensor(sample)


def get_bio_data(from_data_path, to_data_path, ecg_peaks_data_path=None):
    transformer = transforms.Compose([NP_to_Tensor()])
    return BioData(from_data_path, to_data_path, transformer, ecg_peaks_data_path)


def get_data_loader(dataset_prefix, batch_size, from_ppg, shuffle_training=True, shuffle_testing=False):
    train_ppg_data_path = dataset_prefix + 'ppg_train.npy'
    eval_ppg_data_path = dataset_prefix + 'ppg_eval.npy'

    train_ecg_data_path = dataset_prefix + 'ecg_train.npy'
    eval_ecg_data_path = dataset_prefix + 'ecg_eval.npy'

    train_ecg_peaks_data_path = {'opeaks': dataset_prefix + 'ecg_opeaks_train.npy',
                                 'rpeaks': dataset_prefix + 'ecg_rpeaks_train.npy'}

    eval_ecg_peaks_data_path = {'opeaks': dataset_prefix + 'ecg_opeaks_eval.npy',
                                'rpeaks': dataset_prefix + 'ecg_rpeaks_eval.npy'}

    if from_ppg:
        train_data = get_bio_data(
            train_ppg_data_path, train_ecg_data_path, train_ecg_peaks_data_path)
        eval_data = get_bio_data(
            eval_ppg_data_path, eval_ecg_data_path, eval_ecg_peaks_data_path)
    else:
        train_data = get_bio_data(train_ecg_data_path, train_ppg_data_path)
        eval_data = get_bio_data(eval_ecg_data_path, eval_ppg_data_path)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle_training, num_workers=4, pin_memory=True)

    eval_data_loader = DataLoader(
        eval_data, batch_size=15, shuffle=shuffle_testing, num_workers=4, pin_memory=True)

    return train_data_loader, eval_data_loader
