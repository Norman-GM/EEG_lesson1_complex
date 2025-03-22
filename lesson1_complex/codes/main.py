import argparse
from preprocessing import Load_data
from utils import logger
def make_parser():
    parser = argparse.ArgumentParser(description='Lesson1_complex parser')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--dataset_name', type=str, help='dataset name', default='BCICIV2A')
    parser.add_argument('--dataset_path', type=str, help='dataset path', default=r'D:\dataset\BCICIV_2a_gdf')
    parser.add_argument('--use_org_data', type=bool, help='use original data', default=True)
    parser.add_argument('--augmentation', type=bool, help='use augmentation', default=False)
    parser.add_argument('--model', type=str, help='model name', default='EEGNet')
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--seed', type=int, help='random seed', default=2025)
    parser.add_argument('--use_gpu', type=bool, help='use gpu', default=True)
    parser.add_argument('--gpu_id', type=int, help='gpu id', default=0)
    parser.add_argument('--log_dir', type=str, help='log directory', default='./logs')
    parser.add_argument('--save_models_dir', type=str, help='save directory', default='./models')
    parser.add_argument('--save_results_dir', type=str, help='save directory', default='./results')
    parser.add_argument('--save_model', type=bool, help='save model', default=True)
    parser.add_argument("--logger",type=str,help="Logger to be used for metrics.\
     Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",default="wandb"
    )
    return parser

def main():
    args = make_parser().parse_args()
    log = logger(r'results/logs')
    Load_data(args=args, logger=log)


if __name__ == '__main__':
    main()