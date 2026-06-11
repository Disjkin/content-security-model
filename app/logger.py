"""日志配置模块"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "content_security",
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    创建并配置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_format: 日志格式字符串
        log_file: 日志文件路径
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量

    Returns:
        配置完成的 Logger 实例
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(log_format)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "content_security") -> logging.Logger:
    """获取已配置的日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
