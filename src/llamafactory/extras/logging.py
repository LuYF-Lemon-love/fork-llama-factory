# coding:utf-8
#
# src/llamafactory/extras/logging.py
# 
# git pull from LlamaFactory by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jul 24, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Jul 28, 2024
# 
# logging module in LlamaFactory.

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from .constants import RUNNING_LOG


class LoggerHandler(logging.Handler):
    r"""
    Logger handler used in Web UI.
    """

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
        )
        self.setLevel(logging.INFO)
        self.setFormatter(formatter)

        os.makedirs(output_dir, exist_ok=True)
        self.running_log = os.path.join(output_dir, RUNNING_LOG)
        if os.path.exists(self.running_log):
            os.remove(self.running_log)

        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _write_log(self, log_entry: str) -> None:
        with open(self.running_log, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n\n")

    def emit(self, record) -> None:
        if record.name == "httpx":
            return

        log_entry = self.format(record)
        self.thread_pool.submit(self._write_log, log_entry)

    def close(self) -> None:
        self.thread_pool.shutdown(wait=True)
        return super().close()


def get_logger(name: str) -> logging.Logger:
    
    """
    Gets a standard logger with a stream hander to stdout.
    """
    
    # Create a formatter to define the log message format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    
    # Create a stream handler to write log messages to stdout
    handler = logging.StreamHandler(sys.stdout)
    
    # Set the formatter for the handler
    handler.setFormatter(formatter)
    
    # Get or create a logger with the specified name
    logger = logging.getLogger(name)
    
    # Set the logger's logging level to INFO
    logger.setLevel(logging.INFO)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    # Return the created logger
    return logger


def reset_logging() -> None:
    r"""
    Removes basic config of root logger. (unused in script)
    """
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))
