import argparse
import sys
import yaml
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from PIL import Image # 导入 PIL 库，用于读取图片尺寸

# --- 日志基础配置 ---
# 配置日志记录器，使其将日志信息打印到控制台
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

class YoloDatasetSplitter:
    """
    一个专门用于划分YOLO数据集的工具类。
    它负责将匹配的图像和标签文件划分为训练集、验证集和测试集，
    并生成相应的 `data.yaml` 配置文件。
    同时，它会根据图像尺寸将标签中的 x, y, width, height 缩放到 [0, 1] 范围。
    """
    def __init__(self,
                 images_dir: Path,
                 labels_dir: Path,
                 output_dir: Path,
                 classes: List[str],
                 train_rate: float = 0.8,
                 val_rate: float = 0.1):
        """
        初始化YoloDatasetSplitter
        :param images_dir: 原始图像文件所在目录
        :param labels_dir: 原始YOLO格式标签文件(.txt)所在目录
        :param output_dir: 划分后数据集的输出根目录
        :param classes: 数据集中的类别名称列表
        :param train_rate: 训练集所占比例
        :param val_rate: 验证集所占比例
        """
        # --- 输入与输出路径 ---
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir

        # --- 划分比例 ---
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.test_rate = 1.0 - train_rate - val_rate
        self.classes = classes

        # --- 检查参数有效性 ---
        # 检查比例总和是否约等于1.0
        if not abs(self.train_rate + self.val_rate + self.test_rate - 1.0) < 1e-6:
            logger.error("训练集、验证集和测试集的比例之和必须等于1.0，请检查参数。")
            raise ValueError("数据集划分比例之和必须为1.0。")
        # 检查是否提供了类别名称
        if not self.classes:
            logger.error("必须通过 --classes 参数提供类别列表才能生成 data.yaml。")
            raise ValueError("必须提供类别列表。")
        
        # --- 定义输出目录结构 ---
        self.output_dirs: Dict[str, Dict[str, Path]] = {
            "train": {"images": self.output_dir / "train" / "images", "labels": self.output_dir / "train" / "labels"},
            "val": {"images": self.output_dir / "val" / "images", "labels": self.output_dir / "val" / "labels"},
            "test": {"images": self.output_dir / "test" / "images", "labels": self.output_dir / "test" / "labels"}
        }

    def _clean_and_create_dirs(self):
        """清理已存在的输出目录并重新创建所有必需的子目录。"""
        if self.output_dir.exists():
            logger.info(f"发现已存在的输出目录，正在清理: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        logger.info("正在创建新的数据集目录结构...")
        for split in self.output_dirs.values():
            split["images"].mkdir(parents=True, exist_ok=True)
            split["labels"].mkdir(parents=True, exist_ok=True)
        logger.info("目录结构创建成功。")

    def _find_matching_files(self) -> List[Tuple[Path, Path]]:
        """
        在输入目录中查找文件名相同（除扩展名外）的图像和标签文件对。
        :return: 一个包含 (图像路径, 标签路径) 元组的列表。
        """
        txt_files = list(self.labels_dir.glob("*.txt"))
        if not txt_files:
            logger.error(f"在标签目录 '{self.labels_dir}' 中未找到任何 .txt 标签文件。")
            return []

        matching_pairs: List[Tuple[Path, Path]] = []
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        for txt_file in txt_files:
            found_image = False
            for ext in img_extensions:
                image_path = self.images_dir / (txt_file.stem + ext)
                if image_path.exists():
                    matching_pairs.append((image_path, txt_file))
                    found_image = True
                    break
            if not found_image:
                logger.warning(f"未能为标签文件 '{txt_file.name}' 找到匹配的图像，已跳过。")
        
        if not matching_pairs:
            logger.error("未找到任何匹配的图像和标签文件对。请检查文件名称和路径。")
        else:
            logger.info(f"成功找到 {len(matching_pairs)} 对匹配的图像和标签文件。")
        return matching_pairs

    def _normalize_and_write_label(self,
                                   image_path: Path,
                                   original_label_path: Path,
                                   target_label_dir: Path):
        """
        读取原始标签文件，根据图像尺寸归一化坐标，并将归一化后的标签写入新位置。
        原始标签行格式假定为 'class_id x_min y_min width height ...'，但我们在此仅关注 x_min, y_min, width, height。
        或者根据之前的需求，我们现在处理的标签是 '0 x_min y_min width height'
        """
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            logger.error(f"无法读取图片尺寸 '{image_path.name}'：{e}。跳过标签归一化。")
            # 如果无法读取图片尺寸，则直接复制原始标签文件
            # 注意：这里直接复制原始文件可能不符合归一化需求，但为了不中断流程提供后备方案
            # 更好的做法可能是直接抛出错误或跳过该文件对
            shutil.copy(original_label_path, target_label_dir) 
            return

        normalized_lines = []
        try:
            with open(original_label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    # 确保至少有5个部分：class_id, x_min, y_min, width, height
                    if len(parts) >= 5:
                        try:
                            # 根据您之前提供的数据格式，第一个数字是0（class_id），
                            # 接着是 x_min, y_min, width, height
                            class_id = int(parts[0]) # 假设 class_id 是 0
                            x_min = float(parts[1])
                            y_min = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])

                            # 计算中心点坐标
                            center_x = x_min + bbox_width / 2
                            center_y = y_min + bbox_height / 2

                            # 归一化
                            normalized_center_x = center_x / img_width
                            normalized_center_y = center_y / img_height
                            normalized_width = bbox_width / img_width
                            normalized_height = bbox_height / img_height

                            # 格式化为YOLOv5/v8标签格式：class_id center_x center_y width height
                            # 保留5位小数
                            normalized_line = (
                                f"{class_id} {normalized_center_x:.5f} {normalized_center_y:.5f} "
                                f"{normalized_width:.5f} {normalized_height:.5f}"
                            )
                            normalized_lines.append(normalized_line)
                        except ValueError as ve:
                            logger.warning(f"标签行 '{line.strip()}' 格式不正确，无法解析数字：{ve}。跳过该行。")
                    else:
                        logger.warning(f"标签行 '{line.strip()}' 格式不符合预期（少于5个元素）。跳过该行。")

            target_label_path = target_label_dir / original_label_path.name
            with open(target_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(normalized_lines))
        except Exception as e:
            logger.error(f"处理标签文件 '{original_label_path.name}' 时出错: {e}")


    def _copy_files(self, file_pairs: List[Tuple[Path, Path]], split_name: str):
        """
        将文件对（图像和标签）复制到指定的目标集（train/val/test）目录中。
        并对标签文件进行归一化处理。
        :param file_pairs: 要复制的 (图像路径, 标签路径) 列表。
        :param split_name: 目标集的名称 ('train', 'val', or 'test')。
        """
        if not file_pairs:
            return

        target_img_dir = self.output_dirs[split_name]["images"]
        target_label_dir = self.output_dirs[split_name]["labels"]
        
        logger.info(f"正在将 {len(file_pairs)} 个文件复制并处理到 '{split_name}' 数据集...")
        for img_path, label_path in file_pairs:
            try:
                shutil.copy(img_path, target_img_dir) # 复制图片
                self._normalize_and_write_label(img_path, label_path, target_label_dir) # 归一化并写入标签
            except Exception as e:
                logger.error(f"处理文件 {img_path.name} 或 {label_path.name} 时出错: {e}")
        logger.info(f"'{split_name}' 数据集文件复制和处理完成。")

    def _generate_data_yaml(self):
        """
        根据划分结果生成 `data.yaml` 配置文件。
        """
        yaml_path = self.output_dir / "data.yaml"
        
        # 使用相对路径以提高可移植性
        data_yaml_content: Dict[str, Any] = {
            'path': str(self.output_dir.resolve()),  # 使用绝对路径
            'train': str(self.output_dirs['train']['images'].relative_to(self.output_dir)), # 使用相对于 'path' 的相对路径
            'val': str(self.output_dirs['val']['images'].relative_to(self.output_dir)), # 使用相对于 'path' 的相对路径
            'test': str(self.output_dirs['test']['images'].relative_to(self.output_dir)) if self.test_rate > 1e-6 else '',
            'nc': len(self.classes),
            'names': self.classes
        }

        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data_yaml_content, f, default_flow_style=None, sort_keys=False, allow_unicode=True)
            logger.info(f"成功生成数据配置文件: {yaml_path}")
            logger.info(f"配置文件内容:\n{yaml.dump(data_yaml_content, allow_unicode=True)}")
        except Exception as e:
            logger.error(f"生成 data.yaml 文件失败: {e}")

    def run(self):
        """
        执行完整的数据集划分流程。
        """
        logger.info("=== 开始执行YOLO数据集划分流程 ===")
        
        # 0. 清理并创建输出目录
        self._clean_and_create_dirs()
        
        # 1. 查找所有匹配的图像和标签对
        matching_pairs = self._find_matching_files()
        if not matching_pairs:
            logger.critical("没有找到可处理的文件，流程终止。")
            return

        # 2. 划分数据集
        # 第一次划分：划分为训练集和临时集（验证集+测试集）
        train_pairs, temp_pairs = train_test_split(
            matching_pairs,
            train_size=self.train_rate,
            random_state=42, # 使用固定随机种子以保证结果可复现
            shuffle=True
        )
        
        val_pairs, test_pairs = [], []
        # 如果临时集非空，且测试集和验证集都需要划分
        if temp_pairs and self.val_rate > 0 and self.test_rate > 0:
            # 计算验证集在剩余数据中的比例
            val_ratio_in_temp = self.val_rate / (self.val_rate + self.test_rate)
            val_pairs, test_pairs = train_test_split(
                temp_pairs,
                train_size=val_ratio_in_temp,
                random_state=42,
                shuffle=True
            )
        # 如果只有验证集
        elif temp_pairs and self.val_rate > 0:
            val_pairs = temp_pairs
        # 如果只有测试集
        elif temp_pairs:
            test_pairs = temp_pairs

        logger.info("数据集划分完成:")
        logger.info(f"  - 训练集: {len(train_pairs)} 张图片")
        logger.info(f"  - 验证集: {len(val_pairs)} 张图片")
        logger.info(f"  - 测试集: {len(test_pairs)} 张图片")
        
        # 3. 复制文件到各自的目录
        self._copy_files(train_pairs, "train")
        self._copy_files(val_pairs, "val")
        self._copy_files(test_pairs, "test")

        # 4. 生成 data.yaml 配置文件
        self._generate_data_yaml()

        logger.info("=== 数据集划分流程全部完成！ ===")
        logger.info(f"请检查输出目录 '{self.output_dir}' 中的文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO 数据集划分工具。将图像和标签划分为训练、验证和测试集。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--images_dir", type=Path, default='/home/niuoruo/Downloads/WIDER_train/images', help="包含所有原始图像的文件夹路径。")
    parser.add_argument("--labels_dir", type=Path, default='/home/niuoruo/Downloads/wider_face_split/labels', help="包含所有YOLO格式 (.txt) 标签的文件夹路径。")
    parser.add_argument("--output_dir", type=Path, default='/home/niuoruo/workspace/yolo/YOLO/yolo_server/data', help="用于存放划分后数据集的输出文件夹路径。")
    
    parser.add_argument("--classes", type=str, default='face', nargs='+',
                        help="类别名称列表，用空格分隔。例如: --classes person car truck")
    
    parser.add_argument("--train_rate", type=float, default=0.8, help="训练集所占的比例。")
    parser.add_argument("--val_rate", type=float, default=0.1, help="验证集所占的比例。测试集比例将自动计算。")

    args = parser.parse_args()

    try:
        # 创建处理器实例
        splitter = YoloDatasetSplitter(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            classes=args.classes,
            train_rate=args.train_rate,
            val_rate=args.val_rate
        )
        # 运行划分流程
        splitter.run()
    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"发生错误，程序已终止: {e}")
        sys.exit(1)