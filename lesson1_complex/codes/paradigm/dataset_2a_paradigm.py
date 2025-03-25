from codes.dataset.torchdataset import EEG_org_dataset
from sklearn.model_selection import train_test_split
from codes.paradigm.base_paradigm import BaseParadigm
class Dataset_2a_Paradigm(BaseParadigm):
    '''
    Dataset class for the 2a paradigm.
    '''
    def __init__(self, data, label, args):
        super(Dataset_2a_Paradigm, self).__init__(data, label, args)
        self.classes_dict = {
            'left_hand': 0,
            'right_hand': 1,
            'foot': 2,
            'tongue': 3
        }
        self.num_classes = len(self.classes_dict)
        self.sample_rate = 250
        self.window_length = 3
        self.num_channels = 22
        self.num_subject = 9
        self.input_shape = (self.num_channels, self.sample_rate * self.window_length)
        self.output_shape = (self.num_classes, )
    def get_dataset_from_2a(self, sub):
        """
        Get the dataset for the specified paradigm
        """
        if self.args.paradigm == 'cross_session':
            return self.cross_session_split(sub)
        elif self.args.paradigm == 'cross_subject':
            return self.cross_subject_split(sub)
        else:
            raise ValueError(f"Unknown paradigm: {self.args.paradigm}")
    def get_avaliable_paradigm(self):
        '''
        Get the available paradigms for the dataset
        :return:
        '''
        return ['cross_session', 'cross_subject']
    def cross_session_split(self, sub):
        """
        Cross-session paradigm for EEG data
        """
        train_val_data, train_val_label = self.data[sub]['train_data'], self.label[sub]['train_label']
        test_data, test_label = self.data[sub]['test_data'], self.label[sub]['test_label']
        train_data, val_data, train_label, val_label = train_test_split(
            train_val_data, train_val_label, test_size=0.2, random_state=self.args.seed, shuffle=False
        ) # in EEG, to avoid data leakage, we should not shuffle the data
        # Create datasets
        train_dataset = EEG_org_dataset(train_data, train_label, self.args)
        val_dataset = EEG_org_dataset(val_data, val_label, self.args)
        test_dataset = EEG_org_dataset(test_data, test_label, self.args)


        return train_dataset, val_dataset, test_dataset
    def cross_subject_split(self, sub):
        """
        Cross-subject paradigm for EEG data
        """
        pass
