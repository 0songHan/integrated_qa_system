import logging
import os
from .config import Config

def setup_logger(log_file=Config().LOG_FILE):
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    logger = logging.getLogger("EduRAG")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file,encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        # 设置控制台处理器级别
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

print(logger.handlers)


