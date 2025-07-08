import os
import sys
import logging
from typing import Any
import builtins

def setup_logger(output_dir: str, logger) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "project_runtime.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(stream_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Writing logs to {log_file}")


def setup_file_logging(root_dir: str, opened_files) -> None:
    logger = logging.getLogger("project_logger")
    logger.setLevel(logging.INFO)

    _original_open = builtins.open
    def logged_open(file, *args, **kwargs):
        abs_path = os.path.abspath(file)
        if abs_path.startswith(root_dir):
            rel_path = os.path.relpath(abs_path, root_dir)
            if rel_path not in opened_files:
                logger.info(f"Accessed file: {rel_path}")
                opened_files.add(rel_path)
        return _original_open(file, *args, **kwargs)

    builtins.open = logged_open
    return logger