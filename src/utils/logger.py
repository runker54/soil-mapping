import os
import sys
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器，如果日志文件不存在，则创建它"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger