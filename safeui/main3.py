import sys
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QTextEdit, QMessageBox
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import QThread, Signal, Qt
import cv2
import numpy as np
import logging
import argparse # 导入 argparse
from PySide6.QtWidgets import QTextEdit, QScrollBar  # QPlainTextEdit 改为 QTextEdit
from PySide6.QtCore import Qt, Slot, QDateTime
from log_manager import LogManager
# 确保这些模块导入路径正确无误
from yoloside6 import Ui_MainWindow
from utils.infer_stream import stream_inference
from utils.config_utils import load_yaml_config, merger_configs
from utils.beautify import calculate_beautify_params

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SafeUI")

class InferenceThread(QThread):
    frame_ready = Signal(np.ndarray, np.ndarray, object)  # 原始帧, 推理帧, YOLO Result 对象
    progress_updated = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, source, weights, project_args, main_window, yolo_args):
        super().__init__()
        self.source = source
        self.weights = weights
        self.project_args = project_args  # 这是一个 argparse.Namespace 对象
        self.main_window = main_window
        self.yolo_args = yolo_args  # 这是一个 argparse.Namespace 对象
        self.running = True
        self.paused = False
        self.is_camera = source == "0"
        self.is_image = Path(source).is_file() and source.lower().endswith(('.jpg', '.jpeg', '.png'))
        self.is_directory = Path(source).is_dir()
        self.cap = None

    def run(self):
        try:
            total_frames = 0
            frame_interval = 1000 if (self.is_image or self.is_directory) else None
            if self.is_camera or (not self.is_image and not self.is_directory):
                self.cap = cv2.VideoCapture(0 if self.is_camera else self.source)
                if not self.cap.isOpened():
                    self.error_occurred.emit(f"无法打开{'摄像头' if self.is_camera else '视频'}")
                    return
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                frame_interval = 1000 / fps
            else:
                source_path = Path(self.source)
                image_files = [source_path] if self.is_image else sorted(source_path.glob("*.[jp][pn][gf]"))
                total_frames = len(image_files)
                if total_frames == 0:
                    self.error_occurred.emit("目录中无图片文件")
                    return

            # --- 在这里准备 stream_inference 需要的参数 ---
            # 1. 准备 yolo_inference_args (dict)
            yolo_infer_kwargs = vars(self.yolo_args).copy()
            # 用 UI 实时值覆盖 conf 和 iou
            yolo_infer_kwargs['conf'] = self.main_window.ui.conf_num.value()
            yolo_infer_kwargs['iou'] = self.main_window.ui.iou_number.value()
            # 确保 stream 和 show 的设置
            yolo_infer_kwargs['stream'] = True
            yolo_infer_kwargs['show'] = False
            # 从 project_args 中获取保存相关的 flag
            yolo_infer_kwargs['save_txt'] = getattr(self.project_args, 'save_txt', False)
            yolo_infer_kwargs['save_conf'] = getattr(self.project_args, 'save_conf', False)
            yolo_infer_kwargs['save_crop'] = getattr(self.project_args, 'save_crop', False)

            # 2. 准备 project_config (dict)
            project_config_for_stream = vars(self.project_args).copy()
            # --- 参数准备结束 ---

            idx = 0
            # stream_inference 的参数顺序应与定义一致
            for raw_frame, annotated_frame, result in stream_inference(
                weights=self.weights,
                source=self.source,
                project_args=project_config_for_stream, # 使用准备好的字典
                yolo_args=yolo_infer_kwargs, # 使用准备好的完整字典
                pause_callback=lambda: self.paused or not self.running
            ):
                if not self.running:
                    break
                if self.paused:
                    logger.debug("InferenceThread 暂停")
                    self.msleep(100)
                    continue
                self.frame_ready.emit(raw_frame, annotated_frame, result)
                if not self.is_camera:
                    idx += 1
                    progress = int(idx / total_frames * 100) if total_frames > 0 else 0
                    self.progress_updated.emit(progress)
                self.msleep(int(frame_interval) if frame_interval else 10)
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {str(e)}")
            logger.error(f"InferenceThread 错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            logger.info("InferenceThread 已清理")

    def get_yolo_args(self):
        """
        这个方法在推理线程内部调用，每次获取帧时，
        它会从MainWindow的UI控件获取最新的conf和iou值，
        并从InferenceThread自身的project_args (argparse.Namespace)中获取保存相关的flag。
        """
        return {
            'conf': self.main_window.ui.conf_num.value(),
            'iou': self.main_window.ui.iou_number.value(),
            # 修正：使用 getattr() 来访问 argparse.Namespace 对象的属性
            'imgsz': getattr(self.project_args, 'img_width', 640),
            'stream': True,
            'save_txt': getattr(self.project_args, 'save_txt', False),
            'save_conf': getattr(self.project_args, 'save_conf', False),
            'save_crop': getattr(self.project_args, 'save_crop', False),
        }

    def terminate(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        super().terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 1. 初始化 UI 控件：必须最先执行，确保控件可用 ---
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_ui() # 调用自定义的 setup_ui 来进一步配置 UI 控件

        # --- 2. 核心：存储最终合并后的配置参数和临时变量 ---
        self.yolo_args = argparse.Namespace()  # yolo的参数,仅限yolo训练、验证、推理时，官方提供的参数
        self.project_args = argparse.Namespace() # 整个项目的参数
        self.yaml_config = {} # 用于存储加载的YAML配置

        self.inference_thread = None
        self.source = None # 实际选择的输入源文件/目录/摄像头 '0'
        self.model_path = None # 实际选择的模型文件路径
        self.is_camera = False # 初始状态
        self.is_image = False # 初始状态
        self.is_directory = False # 初始状态

        # --- 3. 参数构建与初始化：模拟命令行工具的启动流程 ---

        # 分辨率映射表 (可以从配置文件加载，这里作为默认)
        self.resolution_map = {
            "360": (640, 360),
            "480": (640, 480),
            "720": (1280, 720),
            "1080": (1920, 1080),
            "1440": (2560, 1440),
        }

        # 1. 加载 YAML 配置文件
        self.yaml_config = load_yaml_config(config_type="infer")

        # 2. 从 UI 控件获取**初始**值，并将其封装进 argparse.Namespace 对象
        # 这是为了模拟命令行参数的行为，使 merger_configs 能够正确处理
        initial_ui_args_namespace = argparse.Namespace(
            conf=self.ui.conf_num.value(),
            iou=self.ui.iou_number.value(),
            save=self.ui.save_data.isChecked(),
            model_path=None,  # 模型路径初始为 None，由用户选择后赋值
            source_path=None, # 源路径初始为 None，由用户选择后赋值
            # 如果 yaml_config 中的 use_yaml 默认为 True，则需要在这里设置
            # 如果 merger_configs 内部处理，这里可以不设置
            use_yaml=True # 假设我们总是希望合并 YAML 配置
        )

        # 3. 调用你的 `merger_configs` 函数来合并参数
        # 严格按照你提供的签名进行调用，并传递 argparse.Namespace 对象
        self.yolo_args, self.project_args = merger_configs(
            initial_ui_args_namespace, # 传递 argparse.Namespace 对象
            self.yaml_config,
            mode="infer"
        )

        # 4. 确定显示分辨率并计算美化参数
        # 这里的参数依然从合并后的 self.project_args (argparse.Namespace) 中获取，
        # 通过 getattr() 获取属性值
        display_size_key = str(getattr(self.project_args, 'display_size', '720'))
        display_width, display_height = self.resolution_map.get(display_size_key, self.resolution_map["720"])
        # 设置属性到 project_args
        setattr(self.project_args, 'display_width', display_width)
        setattr(self.project_args, 'display_height', display_height)
        setattr(self.project_args, 'img_width', display_width) # 与 stream_inference 期望的命名保持一致
        setattr(self.project_args, 'img_height', display_height) # 与 stream_inference 期望的命名保持一致

        # 从 project_args 中获取计算美化参数所需的详细设置
        # 这些参数应该已经在 merger_configs 中从 yaml_config 合并到 project_args 中
        # 注意：这里需要确保 calculate_beautify_params 能够接受这些参数，
        # 如果它们在 project_args 中是直接属性，则通过 getattr 访问
        beautify_calculated_params = calculate_beautify_params(
            current_image_height=display_height,
            current_image_width=display_width,
            base_font_size=getattr(self.project_args, 'font_size', 22),
            base_line_width=getattr(self.project_args, 'line_width', 4),
            base_label_padding_x=self.project_args.beautify_settings['base_label_padding_x'],
            base_label_padding_y=self.project_args.beautify_settings['base_label_padding_y'],
            base_radius=self.project_args.beautify_settings['base_radius'],
            ref_dim_for_scaling=720, # 参考尺寸，可以从配置中读取
            font_path=self.project_args.beautify_settings.get('font_path', r"C:\Windows\Fonts\LXGWWenKai-Bold.ttf"), # 从合并后的参数中获取
            text_color_bgr=self.project_args.beautify_settings['text_color_bgr'], # 从合并后的参数中获取
            use_chinese_mapping=self.project_args.beautify_settings['use_chinese_mapping'],
            label_mapping=self.project_args.beautify_settings['label_mapping'], # label_mapping 通常只在yaml中，并被合并到project_args
            color_mapping=self.project_args.beautify_settings['color_mapping']# color_mapping 通常只在yaml中，并被合并到project_args
        )
        setattr(self.project_args, 'beautify_calculated_params', beautify_calculated_params)


        logger.info(f"最终 YOLO 参数: {vars(self.yolo_args)}") # 使用 vars() 打印 Namespace
        logger.info(f"最终项目参数: {vars(self.project_args)}") # 使用 vars() 打印 Namespace
        logger.info(f"计算后的美化参数: {vars(getattr(self.project_args, 'beautify_calculated_params')) if isinstance(getattr(self.project_args, 'beautify_calculated_params'), argparse.Namespace) else getattr(self.project_args, 'beautify_calculated_params')}")

        # --- 5. 根据合并后的最终参数设置 UI 控件的实际初始值 ---
        # 通过 getattr() 获取 Namespace 对象的属性
        self.ui.conf_num.setValue(getattr(self.yolo_args, 'conf', 0.25))
        self.ui.iou_number.setValue(getattr(self.yolo_args, 'iou', 0.45))
        self.ui.save_data.setChecked(getattr(self.project_args, 'save', False))
        # 模型路径和源路径的初始设置
        if getattr(self.project_args, 'model_path', None):
            self.model_path = getattr(self.project_args, 'model_path')
            self.ui.model_name.setText(Path(self.model_path).name)
        if getattr(self.project_args, 'source_path', None):
            self.source = getattr(self.project_args, 'source_path')
            # 根据 source 判断是否是摄像头
            if self.source == "0":
                self.is_camera = True
                self.ui.upload_image.setText("摄像头已选择，点击开始播放")
            elif Path(self.source).is_file():
                self.is_image = self.source.lower().endswith(('.jpg', '.jpeg', '.png'))
                self.show_preview(self.source, is_video=not self.is_image)
            elif Path(self.source).is_dir():
                self.is_directory = True
                self.ui.upload_image.setText(f"已选择目录: {Path(self.source).name}（无图片预览）")


        # --- 6. 连接 UI 信号与槽 ---
        self.connect_signals()

        # 最后更新一次按钮状态，确保 UI 控件值改变后按钮状态正确
        self.update_button_states()

    def project_args_for_thread(self):
        """
        这个方法返回供 InferenceThread 使用的 project_args 对象。
        """
        return self.project_args

    def setup_ui(self):
        # 你的 UI 控件创建和基本属性设置代码
        self.ui.model_name.setReadOnly(True)
        self.ui.model_name.setPlaceholderText("请选择模型文件...")
        self.ui.model_name.setStyleSheet("QLineEdit { border: 1px solid gray; padding: 2px; text-overflow: ellipsis; }")
        self.ui.model_name.setMaximumWidth(200)

        # 确保路径统一为 'yolo_server'
        icon_path = Path(__file__).parent.parent / "yolo_server" / "icons" / "folder.png"
        if icon_path.exists():
            self.ui.model_select.setIcon(QIcon(str(icon_path)))
            self.ui.model_select.setText("")
        else:
            self.ui.model_select.setText("选择模型")

        self.ui.upload_image.setScaledContents(True)
        self.ui.upload_image.setText("上传预览")
        self.ui.finall_result.setScaledContents(True)
        self.ui.finall_result.setText("检测结果")

        self.ui.video_progressBar.setValue(0)
        self.ui.video_progressBar.setTextVisible(True)

        # 初始化视频进度条
        self.ui.video_progressBar.setValue(0)
        self.ui.video_progressBar.setTextVisible(True)  # 进度条显示文本，例如百分比

        # --- 置信度 (Confidence) 设置 ---
        self.ui.conf_num.setRange(0.0, 1.0)
        self.ui.conf_num.setSingleStep(0.05)
        self.ui.conf_num.setValue(0.25)
        self.ui.conf_slider.setMinimum(0)
        self.ui.conf_slider.setMaximum(100)
        self.ui.conf_slider.setValue(25)
        self.ui.iou_number.setRange(0.0, 1.0)
        self.ui.iou_number.setSingleStep(0.05)
        self.ui.iou_number.setValue(0.45)
        self.ui.iou_slider.setMinimum(0)
        self.ui.iou_slider.setMaximum(100)
        self.ui.iou_slider.setValue(45)


        # --- 保存数据选项 ---
        # 保存数据复选框
        self.ui.save_data.setChecked(False)  # **优化：默认不勾选，让用户手动选择是否保存数据**

        self.ui.detection_quantity.setText("未佩戴: 0")
        self.ui.detection_time.setText("0 ms")
        self.ui.detection_result.setText("无检测结果")

        self.statusBar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("FPS: 0")
        self.statusBar.addWidget(self.status_label)
        self.statusBar.addWidget(self.fps_label)

        try:
            from PySide6 import QtWidgets
            if hasattr(self.ui, 'verticalLayout') and isinstance(self.ui.verticalLayout, QtWidgets.QVBoxLayout):
                self.ui.log_display = QTextEdit()
                self.ui.log_display.setReadOnly(True)
                self.ui.log_display.setMaximumHeight(100)
                self.ui.verticalLayout.addWidget(self.ui.log_display)

                class PatchedLogManager(LogManager):
                    def __init__(self, main_window_instance):
                        super().__init__(main_window_instance)
                        self.log_output_text_edit = self.ui.log_display
                        self._setup_log_area()

                    def _setup_log_area(self):
                        self.log_output_text_edit.setReadOnly(True)
                        self.log_output_text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
                        self.log_output_text_edit.setMaximumHeight(100)
                        self.log_output_text_edit.setObjectName("log_display")

                self.log_manager = PatchedLogManager(self)

                class LogManagerHandler(logging.Handler):
                    def __init__(self, log_manager):
                        super().__init__()
                        self.log_manager = log_manager

                    def emit(self, record):
                        self.log_manager.append_log_message(
                            level=record.levelname,
                            source=record.name,
                            message=record.message  # 错误：应为 record.msg
                        )

                logger.handlers.clear()
                logger.setLevel(logging.DEBUG)
                log_handler = LogManagerHandler(self.log_manager)
                log_handler.setLevel(logging.DEBUG)
                logger.addHandler(log_handler)
            else:
                logger.warning("无法添加日志显示，请检查 UI 布局，self.ui.verticalLayout 不存在或不是 QVBoxLayout")
        except Exception as e:
            logger.error(f"日志初始化失败: {type(e).__name__} - {str(e)}")


    def connect_signals(self):
        # 模型和输入源选择按钮
        self.ui.model_select.clicked.connect(self.select_model)
        self.ui.video.clicked.connect(self.select_video)
        self.ui.image.clicked.connect(self.select_image)
        self.ui.dirs.clicked.connect(self.select_dirs)
        self.ui.camera.clicked.connect(self.select_camera)

        # 推理控制按钮
        self.ui.yolo_start.clicked.connect(self.start_inference)
        self.ui.video_start.clicked.connect(self.start_video)
        self.ui.video_stop.clicked.connect(self.stop_video)
        self.ui.video_termination.clicked.connect(self.terminate_video)

        # conf 和 iou 滑块与数字输入框的同步
        self.ui.conf_num.valueChanged.connect(self.sync_conf_slider)
        self.ui.conf_slider.valueChanged.connect(self.sync_conf_num)
        self.ui.iou_number.valueChanged.connect(self.sync_iou_slider)
        self.ui.iou_slider.valueChanged.connect(self.sync_iou_num)

        # 关键：当 'save' 复选框状态改变时，更新 self.project_args
        self.ui.save_data.stateChanged.connect(self._update_save_param_from_ui)

    def sync_conf_slider(self, value):
        self.ui.conf_slider.setValue(int(value * 100))
        logger.debug(f"更新 conf_slider 值: {value}")

    def sync_conf_num(self):
        value = self.ui.conf_slider.value() / 100.0
        self.ui.conf_num.setValue(value)
        logger.debug(f"更新 conf_num 值: {value}")

    def sync_iou_slider(self):
        self.ui.iou_slider.setValue(int(self.ui.iou_number.value() * 100))
        logger.debug(f"更新 iou_slider 值: {self.ui.iou_number.value()}")

    def sync_iou_num(self):
        value = self.ui.iou_slider.value() / 100.0
        self.ui.iou_number.setValue(value)
        logger.debug(f"更新 iou_number 值: {value}")

    def _update_save_param_from_ui(self):
        """
        当 UI 上 'save_data' 复选框状态改变时，更新 self.project_args 中的 'save' 状态，
        并同步更新 self.yolo_args 中控制保存的参数。
        """
        is_checked = self.ui.save_data.isChecked()
        # 更新 project_args 的属性
        setattr(self.project_args, 'save', is_checked)
        # 同步更新 YOLO 推理参数中的保存选项，它们由 project_args['save'] 统一控制
        setattr(self.project_args, 'save_txt', is_checked)
        setattr(self.project_args, 'save_conf', is_checked)
        setattr(self.project_args, 'save_crop', is_checked)
        logger.info(f"UI 更新：保存数据功能设置为: {is_checked}")
        self.update_button_states()

    def select_model(self):
        try:
            # 统一路径为 'yolo_server'
            default_dir = Path(__file__).parent.parent / "yolo_server" / "weights"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择 YOLO 模型文件", str(default_dir), "YOLO 模型文件 (*.pt);;所有文件 (*.*)"
            )
            if file_path:
                self.model_path = file_path
                self.ui.model_name.setText(Path(file_path).name)
                setattr(self.project_args, 'model_path', file_path) # 更新 project_args 的属性
                logger.info(f"选择的模型: {self.model_path}")
            else:
                self.model_path = None
                self.ui.model_name.setText("")
                setattr(self.project_args, 'model_path', None) # 更新 project_args 的属性
                logger.info("未选择模型")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择模型失败")
            logger.error(f"选择模型失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择模型失败: {str(e)}")

    def select_video(self):
        try:
            # 统一路径为 'yolo_server'
            default_dir = Path(__file__).parent.parent / "yolo_server" / "inputs"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", str(default_dir), "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
            )
            if file_path:
                self.terminate_video()
                self.source = file_path
                setattr(self.project_args, 'source_path', file_path) # 更新 project_args 的属性
                self.is_camera = False
                self.is_image = False
                self.is_directory = False
                self.show_preview(file_path, is_video=True)
                logger.info(f"选择的视频: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None) # 更新 project_args 的属性
                logger.info("未选择视频")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择视频失败")
            logger.error(f"选择视频失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择视频失败: {str(e)}")

    def select_image(self):
        try:
            # 统一路径为 'yolo_server'
            default_dir = Path(__file__).parent.parent / "yolo_server" / "inputs"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", str(default_dir), "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*.*)"
            )
            if file_path:
                self.terminate_video()
                self.source = file_path
                setattr(self.project_args, 'source_path', file_path) # 更新 project_args 的属性
                self.is_camera = False
                self.is_image = True
                self.is_directory = False
                self.show_preview(file_path, is_video=False)
                logger.info(f"选择的图片: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None) # 更新 project_args 的属性
                logger.info("未选择图片")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择图片失败")
            logger.error(f"选择图片失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择图片失败: {str(e)}")

    def select_dirs(self):
        try:
            # 统一路径为 'yolo_server'
            default_dir = Path(__file__).parent.parent / "yolo_server" / "inputs"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)

            dir_path = QFileDialog.getExistingDirectory(self, "选择图片或视频目录", str(default_dir))
            if dir_path:
                self.terminate_video()
                self.source = dir_path
                setattr(self.project_args, 'source_path', dir_path) # 更新 project_args 的属性
                self.is_camera = False
                self.is_image = False
                self.is_directory = True
                # 尝试显示目录中第一张图片的预览
                for img_path in Path(dir_path).glob("*.[jp][pn][gf]"):
                    self.show_preview(str(img_path), is_video=False)
                    break
                else:
                    self.ui.upload_image.setText(f"已选择目录: {Path(dir_path).name}（无图片预览）")
                logger.info(f"选择的目录: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None) # 更新 project_args 的属性
                logger.info("未选择目录")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择目录失败")
            logger.error(f"选择目录失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择目录失败: {str(e)}")

    def select_camera(self):
        try:
            self.terminate_video()
            self.source = "0"
            setattr(self.project_args, 'source_path', "0") # 更新 project_args 的属性
            self.is_camera = True
            self.is_image = False
            self.is_directory = False
            self.ui.upload_image.setText("摄像头已选择，点击开始播放")
            logger.info("选择输入: 摄像头")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择摄像头失败")
            logger.error(f"选择摄像头失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择摄像头失败: {str(e)}")

    def show_preview(self, file_path, is_video=False):
        try:
            if is_video:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        self.ui.upload_image.setText("无法读取视频")
                        return
                else:
                    self.ui.upload_image.setText("无法打开视频")
                    return
            else:
                frame = cv2.imread(file_path)
                if frame is None:
                    self.ui.upload_image.setText("无法读取图片")
                    return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.upload_image.setPixmap(pixmap.scaled(self.ui.upload_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logger.debug(f"显示预览: {file_path}, shape: {h}x{w}")
        except Exception as e:
            self.status_label.setText("预览失败")
            logger.error(f"显示预览失败: {str(e)}")
            self.ui.log_display.append(f"错误: 显示预览失败: {str(e)}")

    def start_inference(self):
        try:
            if not self.model_path:
                self.status_label.setText("请先选择模型文件")
                return
            if not self.source:
                self.status_label.setText("请先选择输入源")
                return
            self.start_video()
        except Exception as e:
            self.status_label.setText(f"错误: 开始推理失败")
            logger.error(f"开始推理失败: {str(e)}")
            self.ui.log_display.append(f"错误: 开始推理失败: {str(e)}")

    def start_video(self):
        try:
            if not self.source:
                self.status_label.setText("请先选择输入源")
                self.ui.upload_image.setText("请先选择视频、摄像头、图片或目录")
                return

            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.paused = False
                self.status_label.setText("正在推理")
                logger.info("推理已恢复")
                self.update_button_states()
                return

            self.inference_thread = InferenceThread(
                self.source,
                self.model_path,
                self.project_args_for_thread(), # 传递最终的 project_args (argparse.Namespace)
                self, # 传递 MainWindow 自身的引用
                self.yolo_args
            )
            self.inference_thread.frame_ready.connect(self.update_frames)
            self.inference_thread.progress_updated.connect(self.update_progress)
            self.inference_thread.error_occurred.connect(self.show_error)
            self.inference_thread.finished.connect(self.video_finished)
            self.inference_thread.start()
            self.status_label.setText("正在推理")
            logger.info("推理已开始")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 开始推理失败")
            logger.error(f"开始推理失败: {str(e)}")
            self.ui.log_display.append(f"错误: 开始推理失败: {str(e)}")

    def stop_video(self):
        try:
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.paused = True
                self.status_label.setText("已暂停")
                logger.info("推理已暂停")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 暂停失败")
            logger.error(f"暂停失败: {str(e)}")
            self.ui.log_display.append(f"错误: 暂停失败: {str(e)}")

    def terminate_video(self):
        try:
            logger.info("开始终止线程")
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.running = False
                self.inference_thread.quit()
                self.inference_thread.wait(500) # 等待线程结束，超时500ms
                if self.inference_thread.isRunning(): # 如果线程仍未结束，强制终止
                    self.inference_thread.terminate()
                self.inference_thread = None
                logger.info("推理已终止")
            # 重置 UI 状态
            if not self.is_image: # 如果不是图片，则清空预览
                self.ui.upload_image.setText("上传预览")
            self.ui.finall_result.setText("检测结果")
            self.ui.video_progressBar.setValue(0)
            self.ui.detection_quantity.setText("未佩戴: 0")
            self.ui.detection_time.setText("0 ms")
            self.ui.detection_result.setText("无检测结果")
            self.status_label.setText("就绪")
            self.update_button_states()
            logger.info("UI 已重置")
        except Exception as e:
            self.status_label.setText(f"错误: 停止失败")
            logger.error(f"停止失败: {str(e)}")
            self.ui.log_display.append(f"错误: 停止失败: {str(e)}")

    def closeEvent(self, event):
        try:
            logger.info("开始关闭窗口")
            self.terminate_video()
            event.accept()
            logger.info("窗口已关闭")
        except Exception as e:
            logger.error(f"关闭窗口失败: {str(e)}")
            self.ui.log_display.append(f"错误: 关闭窗口失败: {str(e)}")
            event.ignore()

    def update_frames(self, raw_frame, annotated_frame, result):
        try:
            start_time = cv2.getTickCount()
            # 确保图像尺寸与 QLabel 的尺寸匹配，避免缩放失真或性能问题
            display_width = getattr(self.project_args, 'display_width', self.resolution_map["720"][0])
            display_height = getattr(self.project_args, 'display_height', self.resolution_map["720"][1])

            # 原始帧显示
            raw_frame_resized = cv2.resize(raw_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(raw_frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.upload_image.setPixmap(pixmap.scaled(self.ui.upload_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # 标注帧显示
            annotated_frame_resized = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(annotated_frame_resized, cv2.COLOR_BGR2RGB)
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.finall_result.setPixmap(pixmap.scaled(self.ui.finall_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            no_helmet_count = 0
            total_time = 0.0
            person_count = 0
            safety_helmet_count = 0
            reflective_vest_count = 0
            area_count = 0

            if result is not None and hasattr(result, 'boxes'):
                boxes = result.boxes if result.boxes is not None else []
                # 确保 int(box.cls) 的值在预期的类别范围内
                # 假设类别索引 0: 安全帽, 1: 反光衣, 2: 人员, 3: 未佩戴
                no_helmet_count = sum(1 for box in boxes if hasattr(box, 'cls') and int(box.cls) == 0)
                person_count = sum(1 for box in boxes if hasattr(box, 'cls') and int(box.cls) == 2)
                safety_helmet_count = sum(1 for box in boxes if hasattr(box, 'cls') and int(box.cls) == 4)
                reflective_vest_count = sum(1 for box in boxes if hasattr(box, 'cls') and int(box.cls) == 3)
                area_count = sum(1 for box in boxes if hasattr(box, 'cls') and int(box.cls) == 2) # 假设类别4是常服/区域

                total_time = sum(result.speed.values()) if hasattr(result, 'speed') and result.speed is not None else 0.0

            self.ui.detection_quantity.setText(f"未佩戴: {no_helmet_count}")
            self.ui.detection_time.setText(f"{total_time:.2f} ms")
            self.ui.detection_result.setText(f"""共检测到人员数量有:  {person_count} \n
其中未佩戴安全人员数量有: {no_helmet_count} 个\n
其中佩戴安全帽人员数量有: {safety_helmet_count} 个\n
其中穿着反光衣人员数量有: {reflective_vest_count} 个\n
其中穿戴常服人员数量有: {area_count}\n
当前帧检测耗时: {total_time:.2f} ms
""")

            end_time = cv2.getTickCount()
            frame_time = ((end_time - start_time) / cv2.getTickFrequency()) * 1000
            fps = 1000 / frame_time if frame_time > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.status_label.setText("正在推理")
            logger.debug(f"帧更新耗时: {frame_time:.2f}ms")
        except Exception as e:
            self.status_label.setText("更新帧失败")
            logger.error(f"更新帧失败: {str(e)}")
            self.ui.log_display.append(f"错误: 更新帧失败: {str(e)}")

    def update_progress(self, progress):
        self.ui.video_progressBar.setValue(progress)

    def video_finished(self):
        self.status_label.setText("推理完成")
        logger.info("视频处理完成")
        # 图像处理完成后，不需要重置 UI，除非是目录处理完
        if self.is_image:
             pass # 对于单张图片，不自动清除显示
        else:
            self.terminate_video() # 视频和目录处理完自动终止

    def show_error(self, error_msg):
        self.status_label.setText(f"错误: {error_msg}")
        self.ui.upload_image.setText(error_msg)
        self.ui.finall_result.setText(error_msg)
        self.ui.detection_quantity.setText("未佩戴: 0")
        self.ui.detection_time.setText("耗时: 0 ms")
        self.ui.detection_result.setText("无检测结果")
        self.terminate_video() # 出现错误时终止推理
        logger.error(f"错误: {error_msg}")
        self.ui.log_display.append(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)

    def update_button_states(self):
        # 确保属性已初始化
        has_source = getattr(self, 'source', None) is not None
        has_model = getattr(self, 'model_path', None) is not None
        is_running = bool(getattr(self, 'inference_thread', None) and self.inference_thread.isRunning())
        is_paused = bool(is_running and getattr(self.inference_thread, 'paused', False))
        is_camera_selected = getattr(self, 'is_camera', False)

        # 启动按钮只有在有源、有模型且未运行时才可用
        self.ui.yolo_start.setEnabled(has_source and has_model and not is_running)
        self.ui.video_start.setEnabled(has_source and has_model and (not is_running or is_paused)) # 播放或从暂停恢复
        self.ui.video_stop.setEnabled(is_running and not is_paused) # 暂停按钮只有在运行且未暂停时可用
        self.ui.video_termination.setEnabled(is_running or is_paused) # 终止按钮在运行或暂停时都可用

        # 文件选择按钮在推理进行时禁用
        self.ui.model_select.setEnabled(not is_running and not is_paused)
        self.ui.video.setEnabled(not is_running and not is_paused)
        self.ui.image.setEnabled(not is_running and not is_paused)
        self.ui.dirs.setEnabled(not is_running and not is_paused)
        self.ui.camera.setEnabled(not is_running and not is_paused)

        # 进度条只有在非摄像头模式下才启用
        self.ui.video_progressBar.setEnabled(has_source and not is_camera_selected)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 临时导入 QtWidgets，因为 setup_ui 中可能会用到 QVBoxLayout
    from PySide6 import QtWidgets
    window = MainWindow()
    window.show()
    sys.exit(app.exec())