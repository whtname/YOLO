# Function : 中间层调度器，根据顶层的脚本，决定使用那个数据转换器
import logging
from pathlib import Path
from typing import List, Union

from utils.paths import YOLO_SERVER_ROOT, RAW_DATA_DIR, YOLO_STAGED_LABELS_DIR, ORIGINAL_ANNOTATIONS_DIR

from data_converters.coco import convert_coco_json_to_yolo  # 这是我发的
from data_converters.pascal_voc import convert_pascal_voc_to_yolo

logger = logging.getLogger(__name__)


def convert_data_to_yolo(input_dir: Path,
                         annotation_format: str = "pascal_voc",
                         final_classes_order: Union[List[str], None] = None,
                         coco_task: str = "detection",
                         coco_cls91to80: bool = False
                         ) -> List[str]:
    """
    统一的标注转换入口函数，根据原始标注格式，调用相应的转换器
    :param input_dir: 原始标注文件存放点路径
    :param annotation_format: 原始标注格式，目前支持coco json和pascal voc
    :param final_classes_order: 可选参数，可以指定标签名称列表，指定了为手动模式，不指定为自动模式
    :param coco_task: 针对coco json数据设定的参数，确定任务类别，
    :param coco_cls91to80: coco json特有参数，设定后可以将coco 91种类别映射为80类，一般不做设置
    :return: 返回转换的列表
    """
    logger.info(f"开始处理原始标注数据: ({annotation_format.upper()})格式，数据位于: {input_dir}")

    if not input_dir.exists():
        logger.error(f"原始标注数据不存在，请检查路径: {input_dir}")
        raise FileNotFoundError(f"原始标注数据不存在，请检查路径: {input_dir}")

    classes: List[str] = []

    try:
        if annotation_format == "coco":
            if final_classes_order is not None:
                logger.warning(f"Coco json 转换模式暂不支持手动模式，请使用自动模式，类别将自动从coco json文件中提取")
            classes = convert_coco_json_to_yolo(json_input_dir=input_dir,
                                                 task=coco_task,
                                                 cls91to80=coco_cls91to80)

        elif annotation_format == "pascal_voc":
            logger.info(f"开始将Pascal Voc标注从: '{input_dir.relative_to(YOLO_SERVER_ROOT)}' 转换为YOLO格式")
            classes = convert_pascal_voc_to_yolo(
                xml_input_dir=input_dir,
                output_yolo_txt_dir=YOLO_STAGED_LABELS_DIR,
                target_classes_for_yolo=final_classes_order
            )
            if not classes:
                logger.error(f"Pascal Voc标注转换失败，未提取到任何有效类别标签")
                return []
            logger.info(f"Pascal Voc标注转换成功，已生成YOLO格式标注文件,提取到的类别为: {classes}")
        else:
            logger.error(f"不支持的标注格式: {annotation_format},目前仅支持coco json和pascal_voc")
            raise ValueError(f"不支持的标注格式: {annotation_format},目前仅支持coco json和pascal_voc")
    except Exception as e:
        logger.critical(msg=f"转换过程发生致命错误，转换格式为: {annotation_format}, 错误信息为: {e}", exc_info=True)
        classes = []

    if not classes:
        logger.error(f"转换过程完成，但是未能提取到任何有效的类别，请检查输入数据和转换配置")

    logger.info(f"标注格式: {annotation_format.upper()}转换处理完成，已生成YOLO格式标注文件,提取到的类别为: {classes}")

    return classes


if __name__ == '__main__':
    classes_ = convert_data_to_yolo(input_dir=ORIGINAL_ANNOTATIONS_DIR,
                                    annotation_format="pascal_voc",
                                    coco_task="detection",
                                    coco_cls91to80=True
                                    )
    print(f"{classes_}")
