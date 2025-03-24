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
        data = random_flip(data)
        # You can add more augmentation techniques here
        return data, label