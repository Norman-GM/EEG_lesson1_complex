import numpy as np
from utils import seed_everything

class Load_data():
    def __init__(self, args, logger):
        self.dataset_name = args.dataset_name.upper()
        seed_everything(args.seed)
        self.logger = logger
        self.args = args
        self.data = None
        self.label = None
        self.subjects = None
    # 单例模式
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Load_data, cls).__new__(cls)
        return cls.instance

    def load_data(self):
        if self.dataset_name == 'BCICIV2A':
            from dataset.dataset_class import BCICIV2A
            self.data, self.label = BCICIV2A(self.args).load_org_data()
            self.logger.info(f'Load {self.dataset_name} dataset successfully!')
        elif self.dataset_name == 'numpy_sample':
            # if you want to test the code, you can use this sample data
            self.data = np.random.randn(8, 256,1,64,256)
            self.label = np.random.choice([0, 1], 256)
            self.label = np.repeat(self.label, 8, axis=0)
        return self.data, self.label
