import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from codes.dataset.augmentation import random_flip
class EEG_org_dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for EEG data.

    Args:
        data (numpy.ndarray): EEG data of shape (num_samples, num_channels, num_timepoints).
        labels (numpy.ndarray): Labels corresponding to the EEG data.

    Attributes:
        data (numpy.ndarray): EEG data.
        labels (numpy.ndarray): Labels corresponding to the EEG data.
    """
    def __init__(self, data, labels, args):
        self.data = data
        self.labels = labels
        self.args = args
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.args.augmentation:
            # Apply data augmentation if specified
            data, label = self.augmentation(self.data[idx], self.labels[idx])
        else:
            data, label = self.data[idx], self.labels[idx]
        # Convert data to PyTorch tensor
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        # Return the data and label
        return data, label

    def augmentation(self, data, label):
        '''
        Apply data augmentation techniques to the EEG data.
        '''
        # data = random_flip(data)
        # You can add more augmentation techniques here
        data, label = self.interaug(data, label)
        return data, label

        # Segmentation and Reconstruction (S&R) data augmentation

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label