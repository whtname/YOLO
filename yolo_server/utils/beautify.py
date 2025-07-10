#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName  : beautify.py
# @Time      : 2025/5/26
# @Author    : 雨霓同学
# @Function  : YOLO 检测结果美化绘制（圆角标签、圆角检测框、中英文支持优化，含特殊圆角处理）

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import OrderedDict
import os
import time
import logging

# 获取logger实例
logger = logging.getLogger(__name__)

# ======================= 字体缓存机制 ==========================
# 文本尺寸缓存，存储 (text, font_path, font_size) -> (width, height)
text_size_cache = OrderedDict()
MAX_CACHE_SIZE = 500  # 缓存最大条目数，避免内存溢出


def preload_cache(font_path, font_size, label_mapping):
    """预缓存中英文标签尺寸。

    在这种简化模式下，每个标签只缓存其在给定 font_size 下的尺寸。
    缓存键是标签名，但实际测量时使用带标准化置信度的文本。
    """
    global text_size_cache
    text_size_cache.clear()

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"警告：无法加载字体文件 '{font_path}'。跳过字体大小 {font_size} 的预缓存。")
        return text_size_cache

    # 预缓存中文和英文的典型文本尺寸
    for label_val in list(label_mapping.values()) + list(label_mapping.keys()):
        # 实际用于计算尺寸的文本：带规范化的置信度字符串
        text_to_measure = f"{label_val} 80.0%"

        # 缓存键：只使用标签名
        cache_key = label_val

        temp_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_image)

        if 'font' in locals() and font is not None:
            bbox = draw.textbbox((0, 0), text_to_measure, font=font)
            text_size_cache[cache_key] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        else:
            print(f"警告：字体未成功加载，无法预缓存 '{label_val}'。")
            break
    return text_size_cache


# get_text_size 函数保持不变，与之前修改的版本一致
def get_text_size(text, font_obj, max_cache_size=500):
    """计算文本尺寸（带缓存，缓存键只包含标签）。"""
    parts = text.split(" ")
    if len(parts) > 1 and parts[-1].endswith('%'):
        label_part = " ".join(parts[:-1])
    else:
        label_part = text

    cache_key = label_part

    if cache_key in text_size_cache:
        return text_size_cache[cache_key]

    temp_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_image)
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    text_size_cache[cache_key] = (width, height)

    if len(text_size_cache) > max_cache_size:
        text_size_cache.popitem(last=False)
    return (width, height)



def calculate_beautify_params(
        current_image_height,
        current_image_width,
        base_font_size=26,
        base_line_width=4,
        base_label_padding_x=10,
        base_label_padding_y=10,
        base_radius=8,
        ref_dim_for_scaling=720,
        font_path="LXGWWenKai-Bold.ttf",
        text_color_bgr=(0, 0, 0),
        use_chinese_mapping=True,
        label_mapping=None,
        color_mapping=None
):
    """
    根据图像尺寸动态计算美化参数。

    参数：
        current_image_height (int): 当前图像的高度。
        current_image_width (int): 当前图像的宽度。
        base_font_size (int): 基准字体大小。
        base_line_width (int): 基准线条宽度。
        base_label_padding_x (int): 基准标签水平内边距。
        base_label_padding_y (int): 基准标签垂直内边距。
        base_radius (int): 基准圆角半径。
        ref_dim_for_scaling (int): 参考尺寸（例如720p的720），用于缩放。
        font_path (str): 字体文件的路径。
        text_color_bgr (tuple): 文本颜色 (BGR)。
        use_chinese_mapping (bool): 是否使用中文标签映射。
        label_mapping (dict): 英文标签到中文标签的映射。
        color_mapping (dict): 类别到BGR颜色的映射。
    """
    if label_mapping is None:
        label_mapping = {}
    if color_mapping is None:
        color_mapping = {}

    current_short_dim = min(current_image_height, current_image_width)

    # 防止除以零
    if ref_dim_for_scaling == 0:
        scale_factor = 1.0
        logger.warning("ref_dim_for_scaling 为0，缩放因子将设置为1.0。")
    else:
        scale_factor = current_short_dim / ref_dim_for_scaling

    # 动态调整字体大小和线条宽度，并确保最小尺寸
    font_size_adjusted = max(10, int(base_font_size * scale_factor))
    line_width_adjusted = max(1, int(base_line_width * scale_factor))
    label_padding_x_adjusted = max(5, int(base_label_padding_x * scale_factor))
    label_padding_y_adjusted = max(5, int(base_label_padding_y * scale_factor))
    radius_adjusted = max(3, int(base_radius * scale_factor))  # 圆角半径也进行调整

    # 预加载常用字体尺寸
    # 预加载字体缓存
    cache_dict = preload_cache(font_path, font_size=font_size_adjusted, label_mapping=label_mapping)  # 使用用户提供的字体路径
    print(cache_dict)
    return {
        "font_path": font_path,
        "font_size": font_size_adjusted,
        "line_width": line_width_adjusted,
        "label_padding_x": label_padding_x_adjusted,
        "label_padding_y": label_padding_y_adjusted,
        "radius": radius_adjusted,
        "text_color_bgr": text_color_bgr,
        "use_chinese_mapping": use_chinese_mapping,
        "label_mapping": label_mapping,
        "color_mapping": color_mapping,
    }


# ======================= 绘制辅助函数 ==========================

def draw_filled_rounded_rect(img, pt1, pt2, color, radius,
                             top_left_round=True, top_right_round=True,
                             bottom_left_round=True, bottom_right_round=True):
    """
    绘制一个填充的圆角矩形，可以控制每个角的圆角。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    width = x2 - x1
    height = y2 - y1

    # 绘制矩形的主体部分
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    # 绘制四个角
    if top_left_round:
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x1 + radius, y1 + radius), color, -1)

    if top_right_round:
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x2 - radius, y1), (x2, y1 + radius), color, -1)

    if bottom_left_round:
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1, y2 - radius), (x1 + radius, y2), color, -1)

    if bottom_right_round:
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x2 - radius, y2 - radius), (x2, y2), color, -1)


def draw_bordered_rounded_rect(img, pt1, pt2, color, line_width, radius,
                               top_left_round=True, top_right_round=True,
                               bottom_left_round=True, bottom_right_round=True):
    """
    绘制一个带边框的圆角矩形，可以控制每个角的圆角。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    width = x2 - x1
    height = y2 - y1

    # 绘制直线部分
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, line_width)  # Top
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, line_width)  # Bottom
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, line_width)  # Left
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, line_width)  # Right

    # 绘制圆弧或直线补齐角
    # Top-Left
    if top_left_round:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, line_width)
    else:
        cv2.line(img, (x1, y1 + radius), (x1, y1), color, line_width)  # 垂直线到角
        cv2.line(img, (x1, y1), (x1 + radius, y1), color, line_width)  # 水平线到角

    # Top-Right
    if top_right_round:
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, line_width)
    else:
        cv2.line(img, (x2, y1 + radius), (x2, y1), color, line_width)
        cv2.line(img, (x2 - radius, y1), (x2, y1), color, line_width)

    # Bottom-Left
    if bottom_left_round:
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, line_width)
    else:
        cv2.line(img, (x1, y2 - radius), (x1, y2), color, line_width)
        cv2.line(img, (x1, y2), (x1 + radius, y2), color, line_width)

    # Bottom-Right
    if bottom_right_round:
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, line_width)
    else:
        cv2.line(img, (x2, y2 - radius), (x2, y2), color, line_width)
        cv2.line(img, (x2 - radius, y2), (x2, y2), color, line_width)


# ======================= 自定义美化绘制函数 ==========================

def custom_plot(
        image,
        boxes,
        confs,
        labels,
        use_chinese_mapping=True,
        font_path="LXGWWenKai-Bold.ttf",
        font_size=26,
        line_width=4,
        label_padding_x=10,
        label_padding_y=10,
        radius=8,
        text_color_bgr=(0, 0, 0),
        label_mapping=None,
        color_mapping=None
):
    """
    对检测结果进行美化绘制，包括圆角检测框、圆角标签、中英文支持等。

    参数：
        image: OpenCV 格式的原始图像。
        boxes (np.array): 检测框坐标 (xyxy)。
        confs (np.array): 检测置信度。
        labels (list): 类别标签字符串列表。
        use_chinese_mapping (bool): 是否使用中文标签映射。
        font_path (str): 字体文件的路径。
        font_size (int): 字体大小。
        line_width (int): 线条宽度。
        label_padding_x (int): 标签水平内边距。
        label_padding_y (int): 标签垂直内边距。
        radius (int): 圆角半径。
        text_color_bgr (tuple): 文本颜色 (BGR)。
        label_mapping (dict): 英文标签到中文标签的映射。
        color_mapping (dict): 类别到BGR颜色的映射。

    返回：
        np.array: 绘制完成的图像。
    """
    if label_mapping is None:
        label_mapping = {}
    if color_mapping is None:
        color_mapping = {}

    """绘制检测框和标签 (始终执行美化模式)"""
    result_image_cv = image.copy()  # 先在 OpenCV 图像上进行所有非文本绘制
    img_height, img_width = image.shape[:2]
    try:
        font_pil = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"错误：无法加载字体文件 '{font_path}'。将使用Pillow默认字体。")
        font_pil = ImageFont.load_default()

    # 存储所有需要绘制的文本信息
    texts_to_draw = []

    for box, conf, label_key in zip(boxes, confs, labels):
        x1, y1, x2, y2 = map(int, box)
        color_bgr = color_mapping.get(label_key, (0, 255, 0))  # 默认绿色

        # 标签语言
        if use_chinese_mapping:
            display_label = label_mapping.get(label_key, label_key)
        else:
            display_label = label_key
        label_text_full = f"{display_label} {conf * 100:.1f}%"

        # 计算标签框尺寸和文本尺寸
        text_width, text_height = get_text_size(label_text_full, font_pil, font_size)
        label_box_actual_width = text_width + 2 * label_padding_x
        label_box_actual_height = text_height + 2 * label_padding_y

        # 确保标签框宽度至少能容纳圆角
        label_box_actual_width = max(label_box_actual_width, 2 * radius)

        # 标签框左对齐检测框
        label_box_x_min = int(x1 - line_width // 2)

        # --- 标签位置决策逻辑优化 ---
        # 默认尝试将标签放在检测框上方（外侧）
        label_box_y_min_potential_above = y1 - label_box_actual_height

        # 标记是否在检测框内部绘制标签
        draw_label_inside = False

        # 如果标签框放在上方会超出图像顶部
        if label_box_y_min_potential_above < 0:
            # 尝试将标签放在检测框内部顶部
            if (y2 - y1) >= (label_box_actual_height + line_width * 2):
                label_box_y_min = int(y1 - line_width / 2)
                label_box_y_max = y1 + label_box_actual_height
                draw_label_inside = True
            else:  # 如果检测框太矮，内部也放不下，则放在检测框下方
                label_box_y_min = y2 + line_width
                label_box_y_max = y2 + label_box_actual_height + line_width
                # 检查是否超出图像底部，如果超出则强制贴底
                if label_box_y_max > img_height:
                    label_box_y_max = img_height
                    label_box_y_min = img_height - label_box_actual_height
                draw_label_inside = False
        else:  # 标签可以正常放在检测框上方（外侧）
            label_box_y_min = label_box_y_min_potential_above
            label_box_y_max = y1
            draw_label_inside = False

        # 标签框水平边界检查
        label_box_x_max = label_box_x_min + label_box_actual_width

        # 定义一个标志，指示标签是否需要靠右对齐检测框
        align_right = False
        if label_box_x_max > img_width:
            align_right = True
            label_box_x_min = int(x2 + line_width // 2) - label_box_actual_width  # 标签框右边界与检测框右边界对齐
            if label_box_x_min < 0:  # 如果右对齐后仍然超出左边界，说明标签框比图像宽
                label_box_x_min = 0

        # 判读标签框宽度是否大于检测框宽度 (影响圆角)
        is_label_wider_than_det_box = label_box_actual_width > (x2 - x1)

        # 定义标签框的圆角状态
        label_top_left_round = True
        label_top_right_round = True
        label_bottom_left_round = True
        label_bottom_right_round = True

        # 根据标签位置和对齐方式调整圆角
        if not draw_label_inside:  # 如果标签在框外
            if label_box_y_min == y1 - label_box_actual_height:  # 标签在检测框上方 (外侧)
                if align_right:  # 如果标签靠右对齐检测框
                    label_bottom_left_round = is_label_wider_than_det_box  # 标签左下角圆角，如果标签比检测框宽
                    label_bottom_right_round = False  # 右下角直角，与检测框右上角对齐
                else:  # 标签靠左对齐检测框 (常规情况或超出左边界)
                    label_bottom_left_round = False  # 底部左角直角，与检测框左上角对齐
                    label_bottom_right_round = is_label_wider_than_det_box  # 右下角圆角，如果标签比检测框宽
            elif label_box_y_min == y2 + line_width:  # 标签在检测框下方 (外侧)
                if align_right:  # 如果标签靠右对齐检测框
                    label_top_left_round = is_label_wider_than_det_box  # 标签左上角圆角，如果标签比检测框宽
                    label_top_right_round = False  # 右上角直角
                else:  # 标签靠左对齐检测框
                    label_top_left_round = False  # 顶部左角直角
                    label_top_right_round = is_label_wider_than_det_box  # 右上角圆角，如果标签比检测框宽
        else:  # 如果标签在检测框内部 (上部贴合)
            label_top_left_round = True
            label_top_right_round = True
            if align_right:  # 如果标签在内部且靠右对齐
                label_bottom_left_round = is_label_wider_than_det_box  # 左下角圆角，如果标签比框宽
                label_bottom_right_round = False  # 右下角直角
            else:  # 标签在内部且靠左对齐
                label_bottom_left_round = False
                # 工况 1: 超上边界，标签框宽度小于检测框时，标签框右下角是圆角矩形。
                label_bottom_right_round = is_label_wider_than_det_box or not is_label_wider_than_det_box  # 如果标签在内部，右下角始终圆角
                # 简化为：
                # label_bottom_right_round = True # 因为在内部时，默认其右下角就是圆角

        # 定义检测框的圆角状态 (基于标签位置)
        det_top_left_round = True
        det_top_right_round = True
        det_bottom_left_round = True
        det_bottom_right_round = True

        if not draw_label_inside:  # 如果标签在框外
            if label_box_y_min == y1 - label_box_actual_height:  # 标签在检测框上方
                if align_right:  # 标签靠右对齐检测框
                    det_top_left_round = is_label_wider_than_det_box  # 如果标签比框宽，检测框左上角为圆角
                    det_top_right_round = False  # 检测框右上角直角，与标签框底部对齐
                else:  # 标签靠左对齐检测框
                    det_top_left_round = False  # 检测框左上角直角，与标签框底部对齐
                    # 工况 2 & 3: 正常情况，标签框宽度大于/小于检测框时，检测框右上角圆角/直角
                    det_top_right_round = not is_label_wider_than_det_box  # 如果标签比框宽，右上角直角；否则圆角
            elif label_box_y_min == y2 + line_width:  # 标签在检测框下方
                if align_right:  # 标签靠右对齐检测框
                    det_bottom_left_round = is_label_wider_than_det_box  # 如果标签比框宽，检测框左下角为圆角
                    det_bottom_right_round = False  # 检测框右下角直角
                else:  # 标签靠左对齐检测框
                    det_bottom_left_round = False  # 检测框左下角直角
                    det_bottom_right_round = is_label_wider_than_det_box  # 如果标签比框宽，检测框右下角为圆角
        else:  # 如果标签在检测框内部 (上部贴合)
            det_top_left_round = False
            det_top_right_round = False

        # 绘制检测框 (OpenCV)
        draw_bordered_rounded_rect(result_image_cv, (x1, y1), (x2, y2),
                                color_bgr, line_width, radius,
                                det_top_left_round, det_top_right_round,
                                det_bottom_left_round, det_bottom_right_round)

        # 绘制填充的标签框
        draw_filled_rounded_rect(result_image_cv, (label_box_x_min, label_box_y_min),
                                (label_box_x_min + label_box_actual_width, label_box_y_max),
                                color_bgr, radius,
                                label_top_left_round, label_top_right_round,
                                label_bottom_left_round, label_bottom_right_round)

        # 文本放置在标签框内居中
        text_x = label_box_x_min + (label_box_actual_width - text_width) // 2
        text_y = label_box_y_min + (label_box_actual_height - text_height) // 2

        # 存储文本信息，稍后统一绘制
        texts_to_draw.append({
            'text': label_text_full,
            'position': (text_x, text_y),
            'font': font_pil,
            'fill_bgr': text_color_bgr
        })

    # 统一绘制所有文本
    if texts_to_draw:
        # 将 OpenCV 图像转换为 Pillow 图像，用于文本绘制
        image_pil = Image.fromarray(cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        for text_info in texts_to_draw:
            fill_rgb = (text_info['fill_bgr'][2], text_info['fill_bgr'][1], text_info['fill_bgr'][0])
            draw.text(text_info['position'], text_info['text'], font=text_info['font'], fill=fill_rgb)

        # 将 Pillow 图像转换回 OpenCV 图像
        result_image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

    return result_image_cv