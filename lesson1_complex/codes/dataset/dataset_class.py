from abc import ABC, abstractmethod
import numpy as np
import scipy.io as sio
import h5py
import os
import mne
# 不显示MNE信息
mne.set_log_level('WARNING')
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
                raw.filter(1, 45)
                # raw.notch_filter(50) # in china, the power frequency is 50Hz, but in the US, it is 60Hz,

                # Create the epochs
                if session == 'T':
                    event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10}) if sub != 3 \
                        else dict({'769': 5, '770': 6, '771': 7, '772': 8})
                else:
                    event_id = dict({'783': 7})
                epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=3, baseline=None, preload=True)
                epochs_data = epochs.get_data() * 1e6
                epochs_data = epochs_data.unsqueeze(1) # 288 x 1 x 22 x 876
                epoch_label = epochs.events[:, -1] - min(epochs.events[:, -1]) # 288
                # may use exponential_moving_standardize?
                # epochs_data = self.exponential_moving_standardize(epochs_data)
                # or do some augmentation
                if self.args.augmentation:
                    epochs_data, epoch_label = self.augmentation()
                if session == 'T':
                    self.data.update({sub: {'train_data': epochs_data}})
                    self.label.update({sub: {'train_label': epoch_label}})
                else:
                    self.data[sub].update({'test_data': epochs_data})
                    self.label[sub].update({'test_label': epoch_label})

    def augmentation(self):
        pass


    def exponential_moving_standardize(self, data):
        from braindecode.preprocessing.preprocess import exponential_moving_standardize
        standard_data = exponential_moving_standardize(data)
        return standard_data
