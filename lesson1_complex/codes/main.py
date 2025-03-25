import argparse
from preprocessing import Load_data
from trainer import Trainer
from utils import logger
def make_parser():
    parser = argparse.ArgumentParser(description='Lesson1_complex parser')
    parser.add_argument('--config-dir', type=str, default='configs', help='Directory where config files are stored')
    parser.add_argument('--seed', type=int, help='random seed', default=2025)
    parser.add_argument('--dataset_name', type=str, help='dataset name', default='BCICIV2A')
    parser.add_argument('--dataset_path', type=str, help='dataset path', default=r'D:\dataset\BCICIV_2a_gdf')
    parser.add_argument('--use_org_data', type=bool, help='use original data', default=True)
    parser.add_argument('--paradigm', type=str, help='paradigm', default='cross_session')
    parser.add_argument('--augmentation', type=bool, help='use augmentation', default=False)
    parser.add_argument('--use_gpu', type=bool, help='use gpu', default=True)
    parser.add_argument('--gpu_id', type=int, help='gpu id', default=0)
    parser.add_argument('--log_dir', type=str, help='log directory', default='./logs')
    parser.add_argument('--save_models_dir', type=str, help='save directory', default='./models')
    parser.add_argument('--save_results_dir', type=str, help='save directory', default='./results')
    parser.add_argument('--save_model', type=bool, help='save model', default=True)
    parser.add_argument('--is_choose_best_hyper', type=bool, help='choose best hyperparameters', default=False)
    return parser

def main():
    args = make_parser().parse_args()
    log = logger(r'results/logs')
    load_data_class = Load_data(args=args, logger=log)
    data, label = load_data_class.load_data()
    trainer = Trainer(data, label, args, log)
    trainer.run()

if __name__ == '__main__':
    main()