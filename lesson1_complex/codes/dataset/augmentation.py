import numpy as np

def random_flip(data: np.ndarray) -> np.ndarray:
    """
    Randomly flip the EEG data along the last axis.

    Args:
        data: EEG data of shape (num_samples, num_channels, num_timepoints).

    Returns:
        Flipped EEG data.
    """
    if np.random.rand() > 0.5:
        data = np.flip(data, axis=-1)
    return data

def random_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    Add random Gaussian noise to the EEG data.

    Args:
        data: EEG data of shape (num_samples, num_channels, num_timepoints).
        noise_level: Standard deviation of the Gaussian noise.

    Returns:
        Noisy EEG data.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def random_mask(data: np.ndarray, mask_fraction: float = 0.1) -> np.ndarray:
    """
    Randomly mask a fraction of the EEG data.

    Args:
        data: EEG data of shape (num_samples, num_channels, num_timepoints).
        mask_fraction: Fraction of the data to be masked.

    Returns:
        Masked EEG data.
    """
    mask = np.random.rand(*data.shape) < mask_fraction
    data[mask] = 0
    return data


