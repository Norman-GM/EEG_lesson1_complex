from abc import ABC, abstractmethod
import numpy as np
import scipy.io as sio
import h5py
import os
import mne
from sklearn.preprocessing import StandardScaler
# 不显示MNE信息
mne.set_log_level('ERROR')
# 创建Dataset基类
class Dataset(ABC):
    """
    Abstract base class for datasets.
    """
    def __init__(self, args):
        self.use_augmentation = args.augmentation
        self.dataset_path = args.dataset_path
        pass
    @abstractmethod
    def load_org_data(self):
        """
        Load the dataset.
        """
        pass



class BCICIV2A(Dataset):
    """
    BCICIV2A dataset.
    """
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = 'BCICIV2A'
        self.args = args
        self.data = {}
        self.label = {}

    def load_org_data(self):
        for sub in range(1, 10):
            for session in ['T', 'E']:
                raw = mne.io.read_raw_gdf(os.path.join(self.dataset_path, 'A0' + str(sub) + session + '.gdf'), preload=True)

                raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
                events, annot = mne.events_from_annotations(raw)
                # do some plot if you want
                # mne.viz.plot_events(events, raw.info['sfreq'], raw.first_time, raw.last_time)
                # raw.plot()
                # raw.plot_psd()
                # raw.plot_psd_topo()

                raw.filter(4, 40, fir_design='firwin')
                # raw.notch_filter(50) # in china, the power frequency is 50Hz, but in the US, it is 60Hz,

                # Create the epochs
                if session == 'T':
                    event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10}) if sub != 4 \
                        else dict({'769': 5, '770': 6, '771': 7, '772': 8})
                else:
                    event_id = dict({'783': 7})

                epochs = mne.Epochs(raw, events, event_id, tmin=2, tmax=5.9, baseline=None, preload=True)
                epochs_data = epochs.get_data(copy = True) * 1e6
                for i in range(epochs_data.shape[0]):
                    epochs_data[i] = self.normalize_data(epochs_data[i])
                # may use exponential_moving_standardize?
                # epochs_data = self.exponential_moving_standardize(epochs_data)
                epochs_data = epochs_data[:, np.newaxis, ...][..., 1:] # 288 x 1 x 22 x 751 -> 288 x 1 x 22 x 750
                label_file_name = os.path.join(self.dataset_path, 'true_labels', 'A0' + str(sub) + session + '.mat')
                # load the label
                epoch_label = sio.loadmat(label_file_name)['classlabel'].squeeze() - 1 #  -1 for torch
                # epoch_label = epochs.events[:, -1] - min(epochs.events[:, -1]) # 288  # this way could get train label, but not for test label

                sub_i = sub - 1
                if session == 'T':
                    self.data.update({sub_i: {'train_data': epochs_data}})
                    self.label.update({sub_i: {'train_label': epoch_label}})
                else:
                    self.data[sub_i].update({'test_data': epochs_data})
                    self.label[sub_i].update({'test_label': epoch_label})
        return self.data, self.label
    def normalize_data(self, data):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data
    def exponential_moving_standardize(self, data):
        from braindecode.preprocessing.preprocess import exponential_moving_standardize
        standard_data = exponential_moving_standardize(data)
        return standard_data
