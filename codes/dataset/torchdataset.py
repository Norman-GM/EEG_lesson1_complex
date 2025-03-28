import torch
import torch.nn as nn
import numpy as np
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
        data = self.normalize_data(data) # Normalize the data
        self.args = args
        self.number_class = len(np.unique(labels))
        self.number_seg = 8
        self.trial = data.shape[0]
        self.data_length = data.shape[-1]
        self.number_channel = data.shape[-2]
        if args.augmentation:
            data, labels = self.augmentation(data, labels)
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        # Convert data to PyTorch tensor
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        # Return the data and label
        return data, label
    def normalize_data(self, data):
        """
        Normalize the EEG data
        """
        data = (data - np.mean(data)) / np.std(data)
        return data
    def augmentation(self, data, label):
        '''
        Apply data augmentation techniques to the EEG data.
        '''
        # data = random_flip(data)
        # You can add more augmentation techniques here

        # Segmentation and Reconstruction (S&R) data augmentation
        data, label = self.interaug(data, label)
        return data, label



    def interaug(self, timg, label):

        # https://github.com/snailpt/CTNet/blob/main/Conformer_fine_tune_2a_77.66_2b_85.87.ipynb
        # Here is the original function used in EEGConformer and improved in CTNet
        # if you want to use the original data
        # Add this in __getitem__ method and random choice the
        # same class sample to generate the augmentation data

        aug_data = []
        aug_label = []
        number_records_by_augmentation = self.trial // self.number_class
        number_segmentation_points = self.data_length // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, self.data_length))
            tmp_aug_label = np.zeros((number_records_by_augmentation))
            for trial_i in range(number_records_by_augmentation):
                for seg_i in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[trial_i, :, :, seg_i * number_segmentation_points:(seg_i + 1) * number_segmentation_points] = \
                    tmp_data[rand_idx[seg_i], :, :,seg_i * number_segmentation_points:(seg_i + 1) * number_segmentation_points]
                    tmp_aug_label[trial_i] = clsAug
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_aug_label)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]
        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label