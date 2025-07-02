import logging
import sys
from datetime import datetime
from pathlib import Path
from colorlog import ColoredFormatter
# from pythonjsonlogger import jsonlogger  #  pip install python-json-logger


def setup_logging(base_path: Path,
            log_type: str = "general",
            model_name: str = None,
            log_level: str = logging.INFO,
            temp_log: bool = False,
            encoding: str = "utf-8"
                ):
    """
    配置日志记录器，确保日志存储到指定路径的子目录当中，并同时输出到控制台，
    日志文件名称同时包含类型和时间戳
    :param base_path: 日志文件的根路径
    :param log_type: 日志的类别
    :param model_name: 可选模型名称，用于日志文件名
    :param log_level: 日志记录器最低记录级别
    :param temp_log: 是否启用临时命名(已删除)
    :param encoding: 日志文件的编码格式
    :return: logger 配置好的日志记录器
    """
    # 1. 构建日志文件存放的完整路径
    log_dir = base_path / log_type
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2.生成带有时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 根据temp_log参数决定是否启用临时文件名
    prefix = "temp" if temp_log else log_type.replace(" ", "_")

    # 构建日志文件名，前缀_时间戳_模型名称.log
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "_"))

    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename

    # 3. 获取或创建指定的名称的日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # 阻止日志时间传播到父级logger
    # logger.propagate = False

    # 4. 避免重复添加日志处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    # 5. 创建文件日志处理器
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s"))
    logger.addHandler(file_handler)
    # json_formatter = jsonlogger.JsonFormatter(
    #     "%(asctime)s %(levelname)s %(name)s %(message)s",
    #     json_ensure_ascii=False
    # )
    # file_handler = logging.FileHandler(log_file, encoding=encoding)
    # file_handler.setFormatter(json_formatter)
    # file_handler.setLevel(log_level)
    # logger.addHandler(file_handler)



    # 6. 创建控制台处理器
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s : %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # 7. 输出一些初始化信息
    logger.info(f"日志记录器开始初始化".center(50, "="))
    logger.info(f"日志文件路径: {log_file}")
    logger.info(f"日志记录器初始化时间: {datetime.now()}")
    logger.info(f"日志记录器最低记录级别: {logging.getLevelName(log_level)}")
    logger.info(f"日志记录器初始化完成".center(50, "="))

    return logger


if __name__ == "__main__":
    setup_logging(base_path=Path("."), log_type="test")