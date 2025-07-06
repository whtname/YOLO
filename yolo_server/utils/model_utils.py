# @Function  :模型拷贝
from datetime import datetime
from pathlib import Path
import shutil
import os
import logging

logger = logging.getLogger(__name__)

def copy_checkpoint_models(train_dir_path, model_filename, checkpoint_dir):
    """
    复制训练好的模型到指定的模型归档目录下
    :param train_dir_path:
    :param model_filename:
    :param checkpoint_dir:
    :return:
    """
    if not isinstance(train_dir_path, Path) or not train_dir_path.is_dir():
        logger.error(f"无效的训练目录路径: {train_dir_path}，请传入有效的 pathlib.Path对象目录，跳过模型复制")
        return
    if not isinstance(checkpoint_dir, Path):
        logger.error(f"无效的模型归档目录路径: {checkpoint_dir}，请传入有效的 pathlib.Path对象目录，跳过模型复制")
        return
    # 准备模型保存目录
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"无法创建模型归档目录失败: {e}，请检查目录权限，跳过模型复制")
        return
    # 准备新的模型文件名
    data_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_model_name = Path(model_filename).stem
    train_suffix = train_dir_path.name

    # 遍历并复制模型

    for model_type in ['best', 'last']:
        src_path = train_dir_path / "weights" / f"{model_type}.pt"
        if src_path.exists():
            checkpoint_name = f"{train_suffix}-{data_str}-{base_model_name}-{model_type}.pt"
            dest_path = checkpoint_dir / checkpoint_name
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"模型复制成功: {src_path} -> {dest_path}")
            except FileNotFoundError as e:
                logger.error(f"模型文件不存在: {e}")
            except shutil.SameFileError as e:
                logger.error(f"原文件和目标文件相同: {e}")
            except PermissionError as e:
                logger.error(f"没有权限复制文件: {e}")
            except Exception as e:
                logger.error(f"模型复制发生错误: {e}")
        else:
            logger.warning(f"模型文件不存在: {src_path},跳过文件复制")


if __name__ == "__main__":
    # 运行时获取实际路径信息
    import sys, os, platform

    print("\n===== 环境信息 =====")
    print(f"解释器路径: {sys.executable}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"操作系统: {platform.system()} {platform.release()}")

    run_code = 0
