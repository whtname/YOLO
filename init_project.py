#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName   :init_project.py
# @Time       :2025/6/30 15:38:54
# @Author     :雨觉同学
# @Project    :SafeYolo
# @Function   :用于初始化项目, 创建各种目录结构

import logging

from utils import setup_logging  # 暴露外部接口之后可这样调用
from utils import time_it

from utils.paths import (
    YOLOSERVER_ROOT,
    CONFIGS_DIR,
    DATA_DIR,
    RUNS_DIR,
    MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    CHECKPOINTS_DIR,
    SCRIPTS_DIR,
    LOGS_DIR,
    RAW_DATA_DIR,
    RAW_IMAGES_DIR,
    ORIGINAL_ANNOTATIONS_DIR,
    YOLO_STAGED_LABELS_DIR
)

# 配置日期记录
logger = setup_logging(
    base_path=LOGS_DIR,
    log_type="init_project",
    logger_name="YOLO Init Project",
)

@time_it(iterations=1, name="init_project", logger_instance=logger)
def initialize_project(logger_instance: logging.Logger = None):
    """
    检查并创建项目所有需要的文件夹结构
    :return:
    """
    logger_instance.info("开始初始化项目".center(50, "="))
    logger_instance.info(f"项目的根路径为: {YOLOSERVER_ROOT.resolve()}")

    created_dirs = []
    existed_dirs = []
    raw_data_status = []

    standard_data_to_create = [
        CONFIGS_DIR,
        DATA_DIR,
        RUNS_DIR,
        MODELS_DIR,
        PRETRAINED_MODELS_DIR,
        CHECKPOINTS_DIR,
        SCRIPTS_DIR,
        LOGS_DIR,
        RAW_DATA_DIR,
        RAW_IMAGES_DIR,
        ORIGINAL_ANNOTATIONS_DIR,
        YOLO_STAGED_LABELS_DIR,
        DATA_DIR / "train" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "val" / "images",
        DATA_DIR / "val" / "labels",
        DATA_DIR / "test" / "images",
        DATA_DIR / "test" / "labels",
    ]

    logger_instance.info("开始检查并创建项目核心目录".center(50, "="))
    for d in standard_data_to_create:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                logger_instance.info(f"已成功创建目录: {d.relative_to(YOLOSERVER_ROOT)}")
                created_dirs.append(d.relative_to(YOLOSERVER_ROOT))
            except Exception as e:
                logger_instance.error(f"创建目录 {d.relative_to(YOLOSERVER_ROOT)} 失败: {e}")
                created_dirs.append(f"{d.relative_to(YOLOSERVER_ROOT)} (创建失败 {e})")
        else:
            existed_dirs.append(d.relative_to(YOLOSERVER_ROOT))
            logger_instance.info(f"目录已存在: {d.relative_to(YOLOSERVER_ROOT)}")
    logger_instance.info("项目核心目录检查与创建完毕".center(50, "="))

    # 检查原始数据集目录并给出提示
    logger_instance.info("开始检查原始数据集目录".center(50, "="))
    raw_dirs_to_check = {
        "原始图像文件": RAW_IMAGES_DIR,
        "原始标注文件": ORIGINAL_ANNOTATIONS_DIR
    }
    for desc, raw_dir in raw_dirs_to_check.items():
        if not raw_dir.exists():
            msg = (f"{desc}目录不存在, 请将原始数据放到该路径下, 并确保目录结构正确,以便后续转换数据, "
                   f"输入数据路径为: {raw_dir.resolve()}！")
            logger_instance.warning(msg)
            raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)} (不存在,需要手动创建并放置原始数据)")
        else:
            if not any(raw_dir.iterdir()):
                msg = (f"{desc}目录已经存在,但内容为空, 请将原始{desc}放入到该路径下,以便后续转换数据, "
                       f"输入数据路径为: {raw_dir.relative_to(YOLOSERVER_ROOT)}！")
                logger_instance.warning(msg)
                raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)} (为空,需要放置原始数据)")
            else:
                logger_instance.info(f"原始{desc}目录已存在: {raw_dir.relative_to(YOLOSERVER_ROOT)}包含原始文件")
                raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)} (存在,请勿重复创建)")

    # 汇总所有检查和创建结果
    logger_instance.info("项目初始化结果汇总".center(50, "="))
    if created_dirs:
        logger_instance.info(f"新创建的目录有: {len(created_dirs)}个,具体如下:")
        for d in created_dirs:
            logger_instance.info(f"- {d}")
    else:
        logger_instance.info("本次初始化中,没有创建任何新的目录")
    if existed_dirs:
        logger_instance.info(f"已存在的目录有: {len(existed_dirs)}个,具体如下:")
        for d in existed_dirs:
            logger_instance.info(f"- {d}")
    if raw_data_status:
        logger_instance.info(f"原始数据目录检查结果有: {len(raw_data_status)}个,具体如下:")
        for d in raw_data_status:
            logger_instance.info(f"- {d}")
    logger_instance.info("请务必根据上述提示进行操作,特别是关于原始数据目录的检查结果")
    logger_instance.info("项目初始化完成".center(50, "="))

if __name__ == "__main__":
    initialize_project(logger_instance=logger)
