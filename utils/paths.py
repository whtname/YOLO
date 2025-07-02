from pathlib import Path
YOLOSERVER_ROOT=Path(__file__).resolve().parent.parent

#_核心目录定义
#_配置文件目录
CONFIGS_DIR=YOLOSERVER_ROOT/"configs"
#_数据存放目录
DATA_DIR=YOLOSERVER_ROOT/"data"
#结果存放目录
RUNS_DIR=YOLOSERVER_ROOT/"runs"
#_模型存放目录
MODELS_DIR=YOLOSERVER_ROOT/"models"
#_预训练模型存放目录

#_训练好的模型存放目录
CHECKPOINTS_DIR=MODELS_DIR/"checkpoints"
#_顶层脚本存放目录
SCRIPTS_DIR=YOLOSERVER_ROOT/"scripts"
#_日志文件存放目录
LOGS__DIR=YOLOSERVER_ROOT/"Logs"
#具体数据存放路径
RAW_DATA_DIR=DATA_DIR/"raw"
#_原始图像存放路径
RAW_IMAGES_DIR=RAW_DATA_DIR/"images"
#_原始非Y0L0标注标签存放路径
ORIGINAL_ANNOTATIONS_DIR=RAW_DATA_DIR/"original_annotations"
#_YOL0_txt_数据暂存目录
YOLO_STAGED_LABELS_DIR=RAW_DATA_DIR/"yolo_staged_labels"