#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :infer_frame.py
# @Time      :2025/7/8 09:28:09
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :单帧推理
import cv2
from utils.beautify import custom_plot

def process_frame(frame, result, project_args,beautiful_params, current_fps=None):
    """
    处理单帧图像，完成基本都绘制操作
    :param frame:
    :param result:
    :param project_args:
    :param beautiful_params:
    :param current_fps:
    :return:
    """
    annotated_frame = frame.copy()
    original_height, original_width = frame.shape[:2]

    # 提取boxes, confs, labels
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    labels_idx = result.boxes.cls.cpu().numpy().astype(int)
    labels = [result.names[int(cls_idx)] for cls_idx in labels_idx]

    if project_args.beautify:
        annotated_frame = custom_plot(
            annotated_frame,
            boxes,
            confs,
            labels,
            **beautiful_params
        )
    else:
        annotated_frame = result.plot()
    # 绘制帧率
    if current_fps is not None and current_fps > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 255, 0)
        text_background_color = (0, 0, 0)

        (text_width, text_height), _ = cv2.getTextSize(
            f"FPS: {current_fps:.2f}", font, font_scale, font_thickness)

        padding = 10
        box_x1 = original_width - text_width - padding * 2
        box_y1 = original_height - text_height - padding * 2
        box_x2 = original_width
        box_y2 = original_height

        cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2),
                    text_background_color, -1)
        text_x = original_width - text_width - padding
        text_y = original_height - padding
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (text_x, text_y),
                    font, font_scale, text_color, font_thickness)
    return annotated_frame




if __name__ == "__main__":
    # 运行时获取实际路径信息
    import sys, os, platform

    print("\n===== 环境信息 =====")
    print(f"解释器路径: {sys.executable}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"操作系统: {platform.system()} {platform.release()}")

    run_code = 0
