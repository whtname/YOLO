from typing import Generator, Callable
import logging
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import argparse

# 导入你的utils (确保这些路径在你实际项目中是正确的)
from utils.infer_frame import process_frame
from utils.paths import YOLO_SERVER_ROOT

YOLO_SERVICE_DIR = YOLO_SERVER_ROOT
logger = logging.getLogger("YOLO_Training")

def stream_inference(
        weights: str,
        source: str,
        project_args: dict,
        yolo_args: dict,
        pause_callback: Callable[[], bool] = lambda: False
) -> Generator[tuple[np.ndarray, np.ndarray, object], None, None]:
    """
    执行 YOLOv8 流式推理，并返回原始帧、标注帧和推理结果对象。

    Args:
        weights (str): YOLO 模型权重文件的路径。
        source (str): 推理输入源，可以是图片/视频文件路径、图片目录或摄像头索引 (例如 "0")。
        project_args (dict): 包含所有项目级配置的字典，例如 'display_size', 'beautify' 等。
        yolo_args (dict): 包含 YOLO predict 方法所需参数的字典，如 'conf', 'iou',
                        以及控制保存的 'save_txt', 'save_conf', 'save_crop'。
        pause_callback (Callable[[], bool]): 一个无参数的函数，返回 True 表示暂停推理。

    Yields:
        tuple[np.ndarray, np.ndarray, object]: (原始帧, 标注后的帧, YOLOv8 推理结果对象)。
    """
    logger.info("===== YOLOv8 安全帽检测 UI 推理开始 =====")

    # 打印传入参数以供调试
    print(f"DEBUG - weights: {weights}")
    print(f"DEBUG - source: {source}")
    print(f"DEBUG - project_args: {project_args}")
    print(f"DEBUG - yolo_args: {yolo_args}")

    model = None
    cap = None
    video_writer = None
    last_frame = None
    last_annotated = None

    try:
        # --- 1. 参数准备与路径验证 ---
        resolution_map = {
            360: (640, 360),
            720: (1280, 720),
            1080: (1920, 1080),
            1440: (2560, 1440),
        }
        display_width_set, display_height_set = resolution_map[project_args.get('display_size', 720)]

        beautify_params = {}
        if 'beautify_calculated_params' in project_args:
            beautify_params.update(project_args['beautify_calculated_params'])

        project_args['beautify'] = project_args.get('beautify_settings', {}).get('beautify', False)
        project_args_ns = argparse.Namespace(**project_args)

        model_path = Path(weights)
        if not model_path.is_absolute():
            model_path = YOLO_SERVICE_DIR / "models" / "checkpoints" / weights
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {weights}")

        is_camera_source = source.isdigit()
        if not is_camera_source:
            source_path_obj = Path(source)
            if not source_path_obj.exists():
                logger.error(f"输入源不存在: {source_path_obj}")
                raise FileNotFoundError(f"输入源不存在: {source_path_obj}")
            source = str(source_path_obj)

        logger.info(f"加载模型: {model_path}")
        model = YOLO(str(model_path))

        yolo_args_dict = yolo_args.copy()
        if 'source' in yolo_args_dict:
            del yolo_args_dict['source']
        yolo_args_dict['show'] = False

        # --- 2. 流式推理主循环 ---
        is_video_or_camera = is_camera_source or source.endswith((".mp4", ".avi", ".mov", ".mkv"))
        if is_video_or_camera:
            cap = cv2.VideoCapture(int(source) if is_camera_source else source)
            if not cap.isOpened():
                logger.error(f"无法打开{'摄像头' if is_camera_source else '视频'}: {source}")
                raise RuntimeError(f"无法打开{'摄像头' if is_camera_source else '视频'}: {source}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            current_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"视频分辨率: {current_video_width}x{current_video_height}, 帧率: {fps}")
            logger.info(f"目标显示/保存分辨率: {display_width_set}x{display_height_set}")

            yolo_args_dict['source'] = source
            yolo_args_dict['stream'] = True

            frames_dir = None
            save_dir = None
            video_output_path = None
            idx = 0

            for idx, result in enumerate(model.predict(**yolo_args_dict)):
                if pause_callback():
                    logger.debug("推理暂停")
                    if last_frame is not None:
                        yield last_frame, last_annotated, None
                    cv2.waitKey(100)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                if idx == 0:
                    save_dir = YOLO_SERVICE_DIR / Path(result.save_dir).relative_to(YOLO_SERVICE_DIR)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"推理结果保存目录: {save_dir}")

                    if project_args.get('save_frames', False):
                        frames_dir = save_dir / "0_frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"保存帧图像路径: {frames_dir}")

                    if project_args.get('save', False):
                        video_output_path = save_dir / "output.mp4"
                        video_writer = cv2.VideoWriter(
                            str(video_output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (display_width_set, display_height_set)
                        )
                        if video_writer.isOpened():
                            logger.info(f"视频写入器创建成功，将保存到: {video_output_path}")
                        else:
                            logger.warning(f"视频写入器创建失败，无法保存视频到: {video_output_path}")
                            video_writer = None

                annotated_frame = process_frame(frame, result, project_args_ns, beautify_params)
                display_raw_frame = cv2.resize(frame, (display_width_set, display_height_set))
                display_annotated_frame = cv2.resize(annotated_frame, (display_width_set, display_height_set))

                if video_writer:
                    video_writer.write(display_annotated_frame)
                if frames_dir:
                    frame_path = frames_dir / f"{idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), display_annotated_frame)

                last_frame, last_annotated = display_raw_frame, display_annotated_frame
                yield display_raw_frame, display_annotated_frame, result

            logger.info(f"{'摄像头' if is_camera_source else '视频'}推理完成，结果已保存至: {save_dir or '未保存'}")

        else:  # 处理图片文件或图片目录
            source_path_obj = Path(source)
            is_single_image = source_path_obj.is_file()
            yolo_args_dict['stream'] = not is_single_image  # 目录使用流式推理，单图不使用
            yolo_args_dict['save'] = False  # 禁用 YOLO 内部保存
            yolo_args_dict['source'] = source  # 直接传递目录或单图路径

            save_dir = None
            idx = 0

            # 检查目录是否包含图片（仅用于日志）
            if not is_single_image:
                image_files = sorted(source_path_obj.glob("*.[jp][pn][gf]"))
                if not image_files:
                    logger.error("目录中无图片文件")
                    raise ValueError("目录中无图片文件")

            # 流式推理（目录）或单次推理（单图）
            for idx, result in enumerate(model.predict(**yolo_args_dict)):
                if pause_callback():
                    logger.debug("推理暂停")
                    if last_frame is not None:
                        yield last_frame, last_annotated, None
                    cv2.waitKey(100)
                    continue

                # 读取原始图片
                img_path = Path(result.path) if hasattr(result, 'path') else source_path_obj
                raw_frame = cv2.imread(str(img_path))
                if raw_frame is None:
                    logger.warning(f"无法读取图片: {img_path}，跳过。")
                    continue

                # 初始化保存目录（仅在第一帧/图片）
                if idx == 0 and project_args.get('save', False):
                    if result.save_dir:
                        save_dir = YOLO_SERVICE_DIR / Path(result.save_dir).relative_to(YOLO_SERVICE_DIR)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"推理结果保存目录: {save_dir}")
                    else:
                        logger.warning("YOLO 未返回 save_dir，无法保存结果。")
                        save_dir = None

                annotated_frame = process_frame(raw_frame, result, project_args_ns, beautify_params)
                display_raw_frame = cv2.resize(raw_frame, (display_width_set, display_height_set))
                display_annotated_frame = cv2.resize(annotated_frame, (display_width_set, display_height_set))

                if save_dir and project_args.get('save', False):
                    output_filename = f"{idx:06d}_{img_path.name}" if not is_single_image else img_path.name
                    output_path = save_dir / output_filename
                    cv2.imwrite(str(output_path), display_annotated_frame)
                    logger.debug(f"保存图像: {output_path}")

                last_frame, last_annotated = display_raw_frame, display_annotated_frame
                yield display_raw_frame, display_annotated_frame, result

            logger.info(f"图片推理完成，结果已保存至: {save_dir or '未保存'}")

    except Exception as e:
        logger.error(f"UI 推理失败: {e}", exc_info=True)
        raise
    finally:
        if video_writer:
            video_writer.release()
            logger.info("视频写入器已释放。")
        if cap:
            cap.release()
            logger.info("视频捕获资源已释放。")
        logger.info("===== YOLOv8 安全帽检测 UI 推理结束 =====")