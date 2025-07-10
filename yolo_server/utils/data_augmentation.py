import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
import math

INPUT_IMAGE_DIR = '/home/niuoruo/Downloads/3943f-main/yolo/images'      # 原始图片文件夹路径
INPUT_LABEL_DIR = '/home/niuoruo/Downloads/3943f-main/yolo/labels'      # 原始YOLO标签文件夹路径
OUTPUT_IMAGE_DIR = '/home/niuoruo/Downloads/3943f-main/yolo/augmented_v3/images' # 增强后图片保存路径
OUTPUT_LABEL_DIR = '/home/niuoruo/Downloads/3943f-main/yolo/augmented_v3/labels' # 增强后标签保存路径

USE_IMAGE_BACKGROUND = True                                           # 是否启用随机图片背景
# ！！！请将这里替换为您自己的背景图片文件夹路径！！！
BACKGROUND_IMAGE_DIR = '/home/niuoruo/Downloads/backgrounds'          # 背景图片文件夹路径

CANVAS_WIDTH = 640                      # 新生成画布的宽度
CANVAS_HEIGHT = 640                     # 新生成画布的高度
NUM_AUGMENTED_IMAGES = 1000             # 希望生成多少张增强图片

IMAGES_PER_CANVAS_RANGE = (2, 5)        # 每张新图上粘贴小图的数量范围
SMALL_IMAGE_SIZE_RANGE = (90, 160)      # 小图调整后的大小范围(像素)

# --- 随机旋转 ---
ENABLE_ROTATION = True                  # 是否启用随机旋转
ROTATION_ANGLE_RANGE = (-15, 15)        # 旋转角度范围(度)

# --- 色彩抖动 ---
ENABLE_COLOR_JITTER = True              # 是否启用色彩抖动
BRIGHTNESS_FACTOR_RANGE = (0.7, 1.3)    # 亮度调整因子范围
CONTRAST_FACTOR_RANGE = (0.7, 1.3)      # 对比度调整因子范围
SATURATION_FACTOR_RANGE = (0.7, 1.3)    # 饱和度调整因子范围

def apply_augmentations(img):
    if ENABLE_ROTATION:
        angle = random.uniform(*ROTATION_ANGLE_RANGE)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    if ENABLE_COLOR_JITTER:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(*BRIGHTNESS_FACTOR_RANGE))

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(*CONTRAST_FACTOR_RANGE))

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(*SATURATION_FACTOR_RANGE))

    return img


def create_mosaic_augmentation():
    global USE_IMAGE_BACKGROUND