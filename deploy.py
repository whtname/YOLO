import os
import torch
import numpy as np
import colorsys
import cv2
from datetime import datetime
from ultralytics import YOLO
from utils.efficient_sam import load as load_sam, inference_with_boxes

GOLDEN_RATIO_CONJUGATE = 0.61803398875

def list_models(folder_path):
    return [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.pt')]

def load_model_and_sam(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    sam_model = load_sam(device)
    return model, device, sam_model

def run_detection_on_frame(window):
    # window: MyWindow实例
    if window.cap is None or not window.cap.isOpened():
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 视频源已断开！')
        window.stop_detect()
        return

    ret, frame = window.cap.read()
    if ret:
        window._display_cv_frame(frame, window.oriVideoLabel)

        results = window.model(frame, imgsz=640, conf=window.slider.value()/100, device=window.device)
        annotated_frame = results[0].plot()

        # 根据 segmentation_enabled 决定是否执行分割
        if window.segmentation_enabled:
            masks = inference_with_boxes(
                frame,
                results[0].boxes.xyxy.cpu().numpy(),
                model=window.sam_model,
                device=window.device
            )

            if masks is not None and len(masks) > 0:
                mask_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
                for i, mask in enumerate(masks):
                    hue = (i * GOLDEN_RATIO_CONJUGATE) % 1.0
                    rgb_float = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
                    color = [int(c * 255) for c in rgb_float]
                    mask_bool = mask.astype(bool)
                    mask_overlay[mask_bool] = color

                alpha = 0.4
                final_frame = cv2.addWeighted(mask_overlay, alpha, annotated_frame, 1 - alpha, 0)
            else:
                final_frame = annotated_frame
        else:
            final_frame = annotated_frame

        window._display_cv_frame(final_frame, window.detectlabel)

    else:
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频播放/检测完成！')
        window.stop_detect()

def run_detection_on_file(window):
    # 文件检测（图片/视频）
    file_path = window.file_path
    if not file_path:
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 未选择文件。')
        return

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        frame = cv2.imread(file_path)
        results = window.model(frame, imgsz=640, conf=window.slider.value()/100, device=window.device)
        annotated_frame = results[0].plot()

        if window.segmentation_enabled:
            masks = inference_with_boxes(
                frame,
                results[0].boxes.xyxy.cpu().numpy(),
                model=window.sam_model,
                device=window.device
            )
            if masks is not None and len(masks) > 0:
                mask_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
                for i, mask in enumerate(masks):
                    hue = (i * GOLDEN_RATIO_CONJUGATE) % 1.0
                    rgb_float = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
                    color = [int(c * 255) for c in rgb_float]
                    mask_bool = mask.astype(bool)
                    mask_overlay[mask_bool] = color
                alpha = 0.4
                final_frame = cv2.addWeighted(mask_overlay, alpha, annotated_frame, 1 - alpha, 0)
            else:
                final_frame = annotated_frame
        else:
            final_frame = annotated_frame

        window._display_cv_frame(final_frame, window.detectlabel)
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 检测完成。')

    elif file_extension in ['.mp4', '.avi']:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 无法打开视频文件！')
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = window.model(frame, imgsz=640, conf=window.slider.value()/100, device=window.device)
            annotated_frame = results[0].plot()
            if window.segmentation_enabled:
                masks = inference_with_boxes(
                    frame,
                    results[0].boxes.xyxy.cpu().numpy(),
                    model=window.sam_model,
                    device=window.device
                )
                if masks is not None and len(masks) > 0:
                    mask_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
                    for i, mask in enumerate(masks):
                        hue = (i * GOLDEN_RATIO_CONJUGATE) % 1.0
                        rgb_float = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
                        color = [int(c * 255) for c in rgb_float]
                        mask_bool = mask.astype(bool)
                        mask_overlay[mask_bool] = color
                    alpha = 0.4
                    final_frame = cv2.addWeighted(mask_overlay, alpha, annotated_frame, 1 - alpha, 0)
                else:
                    final_frame = annotated_frame
            else:
                final_frame = annotated_frame
            window._display_cv_frame(final_frame, window.detectlabel)
            cv2.waitKey(1)
        cap.release()
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频检测完成。')
    else:
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 不支持的文件类型。')

def start_camera(window):
    # 打开摄像头
    if window.cap is not None:
        window.cap.release()
    window.cap = cv2.VideoCapture(0)
    if not window.cap.isOpened():
        window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 无法打开摄像头！')
        window.cap = None
        return
    window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 摄像头已打开。')
    window.timer.start(30)
    window.startDetectBtn.setEnabled(False)
    window.openFileBtn.setEnabled(False)
    window.stopDetectBtn.setEnabled(True)

def stop_camera(window):
    # 停止摄像头/检测
    if window.cap is not None:
        window.cap.release()
        window.cap = None
    window.timer.stop()
    window.startDetectBtn.setEnabled(True)
    window.openFileBtn.setEnabled(True)
    window.stopDetectBtn.setEnabled(False)
    window.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 检测已停止。')