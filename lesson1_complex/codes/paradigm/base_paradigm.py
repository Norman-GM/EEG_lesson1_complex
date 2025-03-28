from abc import ABC, abstractmethod
class BaseParadigm(ABC):
    """
    Abstract base class for paradig
    """

    def __init__(self, data, label, args):
        """
        Initialize the base class for one subject.
        :param data: EEG data
        :param label: EEG labels
        """
        self.data = data
        self.label = label
        self.args = args

    @abstractmethod
    def get_dataset_from_2a(self, sub):
        """
        Get the dataset for the specified paradigm
        """
        pass

    @abstractmethod
    def get_avaliable_paradigm(self):
        '''
        Get the available paradigms for the dataset
        :return:
        '''
        pass