import logging
import time
from functools import wraps


_default_logger = logging.getLogger(__name__)

def time_it(iterations: int = 1, name: str = None, logger_instance: logging.Logger = _default_logger):
    """
    性能测量工具
    :param iterations:迭代次数
    :param name:函数名称,如果为None.使用装饰器的原名称
    :param logger instance:日志对象
    :return:
    """

    _logger_to_use = logger_instance if logger_instance is not None else default_logger
    # 辅助函数，根据总秒数格式化为最合适的到单位
    def _format_time_auto_unit(total_seconds: float) -> str:
        if total_seconds < 0.000001:
            return f"{total_seconds *1_000_000:.3f} 纳秒"
        elif total_seconds < 0.001:
            return f"{total_seconds *1_000:.3f} 微秒"
        elif total_seconds < 1.0:
            return f"{total_seconds * 1000:.3f} 亳秒"
        elif total_seconds < 60: # 小于1分钟
            return f"{total_seconds:.2f} 秒"
        elif total_seconds < 3600:# 小于1小时
            minutes =total_seconds // 60
            seconds = total_seconds %60
            return f"{minutes:.0f} 分钟 {seconds:.2f} 秒"
        else:
            hours = total_seconds // 3600
            minutes =(total_seconds % 3600)// 60
            seconds =(total_seconds %3600)%60
            return f"{hours:.0f} 小时 {minutes:.0f} 分钟 {seconds:.2f} 秒"
    
    def decorator(func):
    
            @wraps(func)#保留原函数的元信息
            def wrapper(*args, **kwargs):
                func_display_name = name if name is not None else func.__name__
                total_elapsed_time = 0.0
                result = None

                for i in range( iterations):
                    start_time = time.perf_counter()# 获取高精度时间
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()
                    total_elapsed_time += end_time - start_time
                avg_elapsed_time = total_elapsed_time /iterations
                # 使用辅助函数格式化平均耗时
                formatted_avg_time=_format_time_auto_unit(avg_elapsed_time)

                if iterations == 1:
                    _logger_to_use.info(f"性能报告:'{func_display_name}'平均耗时:{formatted_avg_time}")
                else:
                    _logger_to_use.info(f"性能报告:'{func_display_name}'执行了:{iterations} 次,"
                                        f"平均耗时:{formatted_avg_time}")
                return result
            return wrapper
    return decorator


if __name__ == "__main__":
    from utils.logging_utils import setup_logging
    from pathlib import Path
    logger = setup_logging(base_path=Path("."),log_type="test")
    
    @time_it(iterations=5,name="test func",logger_instance=logger)

    def test_func():
        time.sleep(0.5)
    test_func()