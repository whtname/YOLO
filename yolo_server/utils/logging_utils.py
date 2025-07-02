import logging
from datetime import datetime
from pathlib import Path


def setup_logging(base_path: Path,
                log_type:str = "general",
                model_name:str=None,
                log_level:str=logging.info,
                temp_log:bool=False,
                logger_name:str='YOLO DEFAULT',
                encoding:str='utf-8'
                    ):
    """
    """

    log_dir = base_path / log_type
    log_dir.mkdir(parents=True,exsit_ok=True)

    timestamp=datetime.now().strftime("%Y%M%D-%H%M%S")
    prefix = "temp" if temp_log else log_type.replace(" ","_")

    log_filename_parts = [prefix,timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ","_"))
