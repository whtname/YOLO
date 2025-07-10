#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_infer_v3.py
# @Time      :2025/7/8 09:37:36
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :集成之前所有的功能+添加美化功能
import argparse

import cv2
from ultralytics import YOLO
from pathlib import Path

import sys
yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0,str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1,str(utils_path))

from utils.logging_utils import setup_logging
from utils.config_utils import load_yaml_config, merger_configs, log_parameters
from utils.paths import LOGS_DIR, CHECKPOINTS_DIR, YOLO_SERVER_ROOT
from utils.beautify import calculate_beautify_params
from utils.infer_frame import process_frame
from utils.tts_utils import process_tts_detection, init_tts


def parser_args():
    parser = argparse.ArgumentParser(description="工地安全生产检测系统推理脚本")
    parser.add_argument("--model", type=str,
        default=r"train39-20250707-135707-yolo11n-best.pt",
        help="模型权重文件")
    parser.add_argument("--source", type=str,
                default=r"0", help="推理图片")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--save", type=bool, default=True, help="是否保存推理结果")
    parser.add_argument("--show", type=bool, default=True, help="是否显示推理结果")
    parser.add_argument("--save_txt", type=bool, default=True, help="是否保存推理结果txt文件")
    parser.add_argument("--save_conf", type=bool, default=True, help="是否保存推理结果置信度")
    parser.add_argument("--save_crop", type=bool, default=True, help="是否保存推理结果裁剪图片")
    parser.add_argument("--save_frames", type=bool, default=True, help="是否保存推理结果帧")

    # 美化参数
    parser.add_argument("--display_size", type=str,default="720",
                    choices=["360","480","720","1080","1440"], help="显示图片大小")
    parser.add_argument("--beautify", type=bool, default=True, help="是否美化推理结果")
    parser.add_argument("--use_chinese_mapping", type=bool, default=True, help="是否使用中文映射")
    parser.add_argument("--font_size", type=int, default=22, help="字体大小")
    parser.add_argument("--line_width", type=int, default=4, help="边框宽度")
    parser.add_argument("--label_padding_x", type=int, default=5, help="标签内边距X")
    parser.add_argument("--label_padding_y", type=int, default=5, help="标签内边距Y")
    parser.add_argument("--radius", type=int, default=4, help="边框圆角半径")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用yaml配置文件")
    # 语音合成引擎内容
    parser.add_argument("--tts_enable", type=bool, default=True, help="是否启用语音合成")
    parser.add_argument("--tts_duration", type=int, default=10, help="语音合成时长")
    parser.add_argument("--tts_interval", type=int, default=10, help="语音间隔")
    parser.add_argument("--tts_text", type=str,
                        default="您已进入施工场所,请规范佩戴安全帽!!!", help="语音合成内容")

    return parser.parse_args()

def main():
    args = parser_args()

    # 1. 设置日志
    logger = setup_logging(
        base_path= LOGS_DIR,
        log_type="infer",
        model_name=args.model,
        temp_log=False
    )

    # 2. 打印设备信息
    # 3. 加载YAML配置文件
    yaml_config = {}
    if args.use_yaml:
        yaml_config = load_yaml_config(config_type="infer")
    # 4. 合并配置文件
    yolo_args, project_args = merger_configs(args, yaml_config, mode="infer")

    # 5. 分辨率映射
    resolution_map = {
        "360": (640, 360),
        "480": (640, 480),
        "720": (1280, 720),
        "1080": (1920, 1080),
        "1440": (2560, 1440),
    }
    display_width, display_height = resolution_map[args.display_size]

    # 6. 计算美化参数
    beautiful_params = calculate_beautify_params(
        current_image_height=display_height,
        current_image_width=display_width,
        base_font_size=args.font_size,
        base_line_width=args.line_width,
        base_label_padding_x=args.label_padding_x,
        base_label_padding_y=args.label_padding_y,
        base_radius=args.radius,
        ref_dim_for_scaling=720,
        font_path="LXGWWenKai-Bold.ttf",
        text_color_bgr=(0, 0, 0),
        use_chinese_mapping=args.use_chinese_mapping,
        label_mapping=yaml_config['beautify_setting']['label_mapping'],
        color_mapping=yaml_config["beautify_settings"]['color_mapping']
    )
    # 初始TTS语音合成引擎
    tts_engine = init_tts() if args.tts_enable else None
    if args.tts_enable and not tts_engine:
        logger.error("TTS语音合成引擎初始化失败")
        args.tts_enable = False
    tts_state = {
        'no_helmet_start_time': None,
        'last_tts_time': None,
    }
    tts_text = args.tts_text


    # 7. 打印一下参数来源信息
    log_parameters(project_args)
    # 8. 加载模型
    model = YOLO(CHECKPOINTS_DIR / args.model)
    logger.info(f"模型加载成功模型: {CHECKPOINTS_DIR / args.model}")
    source = args.source
    # 模型推理
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # 初始化视频捕获
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            raise RuntimeError(f"无法打开视频源: {source}")
        window_name = "YOLOv8 Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 流式推理
        video_writer = None
        frames_dir = None
        yolo_args.stream = True
        yolo_args.show = False
        yolo_args.save = False
        print('YOLO参数',yolo_args)
        for idx, result in enumerate(model.predict(**vars(yolo_args))):
            # 第一帧初始化保存路径
            if idx == 0:
                save_dir = YOLO_SERVER_ROOT / Path(result.save_dir)
                logger.info(f"此次推理结果保存路径: {save_dir}")
                if args.save_frames:
                    frames_dir = save_dir / "0_frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"保存帧图像路径: {frames_dir}")
                if args.save:
                    video_path = save_dir / "output.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (display_width, display_height),
                    )
                    logger.info(f"保存视频路径: {video_path}")
                    if video_writer:
                        logger.info(f"视频写入器创建成功")
            # 获取每一帧
            frame = result.orig_img
            # 处理TTS
            process_tts_detection(result, args.tts_enable, args.tts_duration,
                                args.tts_interval,tts_engine,tts_state,tts_text
                                )

            # 针对每一帧进行美化
            annotated_frame = process_frame(frame, result, project_args,beautiful_params)

            # 保存视频
            if video_writer:
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
                video_writer.write(annotated_frame)
            # 保存帧图像
            if frames_dir:
                cv2.imwrite(str(frames_dir / f"{idx}.png"), annotated_frame)

            # 显示
            cv2.imshow(window_name, annotated_frame)

            # 退出机制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
    else:
        yolo_args.stream = False
        yolo_args.show = False
        results = model.predict(**vars(yolo_args))
        save_dir = Path(results[0].save_dir)
        base_save_dir = save_dir / "beautify"
        base_save_dir.mkdir(parents=True, exist_ok=True)
        for ids, result in enumerate(results):
            annotated_frame = process_frame(result.orig_img, result, project_args,beautiful_params)
            if args.save:
                save_path = base_save_dir / f"{ids}.png"
                cv2.imwrite(str(save_path), annotated_frame)
    logger.info(f"推理结束".center(50, "="))



if __name__ == "__main__":
    main()
