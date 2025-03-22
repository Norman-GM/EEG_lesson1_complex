import numpy as np
import random
import torch
import logging
import os
import sys
import shutil
from datetime import datetime

def logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # 如果文件夹内的log文件超过20个， 则删除最早的一个
    log_files = [f for f in os.listdir(os.path.dirname(log_path)) if f.endswith('.log')]
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(os.path.dirname(log_path), x)))
    if len(log_files) > 20:
        os.remove(os.path.join(os.path.dirname(log_path), log_files[0]))
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler， 按照当前时间生成的日志文件
    log_path = os.path.join(log_path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setStream(sys.stdout)
    # 定义handler的输出格式
    # 设置颜色
    class ColorFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.ERROR:
                record.msg = '\033[1;31m' + record.msg + '\033[0m'
            elif record.levelno == logging.WARNING:
                record.msg = '\033[1;33m' + record.msg + '\033[0m'
            elif record.levelno == logging.INFO:
                record.msg = '\033[1;34m' + record.msg + '\033[0m'
            return logging.Formatter.format(self, record)
    class PlainFormatter(logging.Formatter):
        def format(self, record):
            return logging.Formatter.format(self, record)
    formatter = ColorFormatter(' %(message)s')
    PlainFormatter = PlainFormatter('%(message)s')
    file_handler.setFormatter(PlainFormatter)
    console_handler.setFormatter(formatter)
    # 给logger添加handler
    # 如果是debug模式， 则不保存日志
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"## **日志记录时间为 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    return logger
def seed_everything(seed: int = 2025):
    """
    Set the random seed for reproducibility.
    Args:
        seed
            (int): The seed value to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
