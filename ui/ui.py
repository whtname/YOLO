import os
from datetime import datetime
import cv2
import torch
import numpy as np
import random
import colorsys
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QDir
from PySide6.QtGui import QIcon
from ultralytics import YOLO
from utils.paths import CHECKPOINTS_DIR
from utils.efficient_sam import load, inference_with_boxes

GOLDEN_RATIO_CONJUGATE = 0.61803398875

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- 数据成员初始化 ---
        self.model = None
        self.cap = None
        self.video = None
        self.file_path = None
        self.base_name = None
        self.folder_path = CHECKPOINTS_DIR
        self.segmentation_enabled = False 

        # --- 定时器初始化 ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.detect_frame)

        # --- 初始化UI ---
        self.init_gui()

    def init_gui(self):
        """初始化主窗口和整体布局"""
        self.setFixedSize(1400, 850) 
        self.setWindowTitle('目标检测与分割')
        self.setWindowIcon(QIcon("logo.jpg")) 

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        
        self.set_background_image('./ui/bg.png')

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20) 

        display_layout = self._create_display_layout()
        main_layout.addLayout(display_layout)

        control_layout = self._create_control_layout()
        main_layout.addLayout(control_layout)

        main_layout.addStretch() 

    def _create_display_layout(self):
        """创建顶部用于显示视频和结果的布局"""
        display_layout = QtWidgets.QHBoxLayout()
        display_layout.setSpacing(25)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(15)

        self.oriVideoLabel = QtWidgets.QLabel("原始图像")
        self.oriVideoLabel.setFixedSize(600, 450)
        self.oriVideoLabel.setAlignment(Qt.AlignCenter)
        self.oriVideoLabel.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc; 
                border-radius: 10px; 
                background-color: rgba(255, 255, 255, 0.6);
                font-size: 16px;
                font-family: "Microsoft YaHei";
            }
        """)
        left_layout.addWidget(self.oriVideoLabel)

        self.outputField = QtWidgets.QTextBrowser()
        self.outputField.setFixedSize(600, 150)
        self.outputField.setStyleSheet("""
            QTextBrowser {
                border: 2px solid #ccc;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.8);
                font-family: "Consolas", "Courier New", "Microsoft YaHei";
                font-size: 13px;
                padding: 5px;
            }
        """)
        left_layout.addWidget(self.outputField)
        left_layout.addStretch()

        self.detectlabel = QtWidgets.QLabel("检测结果")
        self.detectlabel.setFixedSize(650, 615)
        self.detectlabel.setAlignment(Qt.AlignCenter)
        self.detectlabel.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc; 
                border-radius: 10px; 
                background-color: rgba(255, 255, 255, 0.6);
                font-size: 16px;
                font-family: "Microsoft YaHei";
            }
        """)

        display_layout.addLayout(left_layout)
        display_layout.addWidget(self.detectlabel)
        
        return display_layout

    def _create_control_layout(self):
        """创建底部用于控制操作的布局"""
        control_layout = QtWidgets.QHBoxLayout()
        
        groupbox_style = """
            QGroupBox {
                font-size: 14px;
                font-family: "Microsoft YaHei";
                border: 1px solid gray;
                border-radius: 9px;
                margin-top: 10px;
                background-color: rgba(255, 255, 255, 0.5);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """
        button_style = """
            QPushButton {
                background-color: white;
                border: 1px solid gray;
                border-radius: 8px;
                padding: 5px;
                font-size: 14px;
                font-family: "Microsoft YaHei";
                min-height: 40px;
                min-width: 110px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #888888;
            }
        """

        settings_group = QtWidgets.QGroupBox("")
        settings_group.setStyleSheet(groupbox_style)
        settings_layout = QtWidgets.QVBoxLayout(settings_group)
        settings_layout.setSpacing(5)
        
        model_layout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QComboBox()
        self.selectModel.setMinimumHeight(40)
        self.selectModel.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei";')
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.pt'):
                self.selectModel.addItem(os.path.splitext(filename)[0])
        self.loadModel = QtWidgets.QPushButton('🔄️ 加载模型')
        self.loadModel.setStyleSheet(button_style)
        self.loadModel.clicked.connect(self.load_model)
        model_layout.addWidget(self.selectModel, 2)
        model_layout.addWidget(self.loadModel, 1)

        conf_layout = QtWidgets.QHBoxLayout()
        self.con_label = QtWidgets.QLabel('置信度:')
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(50)
        self.spinbox = QtWidgets.QDoubleSpinBox()
        self.spinbox.setRange(0.01, 0.99)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setValue(0.5)
        self.spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.slider.valueChanged.connect(lambda v: self.spinbox.setValue(v / 100.0))
        self.spinbox.valueChanged.connect(lambda v: self.slider.setValue(int(v * 100)))
        conf_layout.addWidget(self.con_label)
        conf_layout.addWidget(self.slider)
        conf_layout.addWidget(self.spinbox)
        
        self.confidence_widget = QtWidgets.QWidget()
        self.confidence_widget.setLayout(conf_layout)
        self.confidence_widget.setEnabled(False)

        self.segmentation_checkbox = QtWidgets.QCheckBox("启用图像分割")
        self.segmentation_checkbox.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei"; margin-left: 5px;')
        self.segmentation_checkbox.setEnabled(False) 
        self.segmentation_checkbox.stateChanged.connect(self.toggle_segmentation)

        settings_layout.addLayout(model_layout)
        settings_layout.addWidget(self.confidence_widget)
        settings_layout.addWidget(self.segmentation_checkbox)
        settings_layout.addStretch()

        actions_group = QtWidgets.QGroupBox("")
        actions_group.setStyleSheet(groupbox_style)
        actions_layout = QtWidgets.QHBoxLayout(actions_group)
        actions_layout.setSpacing(5)

        self.openFileBtn = QtWidgets.QPushButton('🖼️ 上传文件')
        self.openFileBtn.setStyleSheet(button_style)
        self.openFileBtn.clicked.connect(self.upload_file)
        self.openFileBtn.setEnabled(False)

        self.startDetectBtn = QtWidgets.QPushButton('🔍 开始检测')
        self.startDetectBtn.setStyleSheet(button_style)
        self.startDetectBtn.clicked.connect(self.show_detect)
        self.startDetectBtn.setEnabled(False)
        
        self.startCameraBtn = QtWidgets.QPushButton('📹 打开摄像头')
        self.startCameraBtn.setStyleSheet(button_style)
        self.startCameraBtn.clicked.connect(self.start_camera_detect)
        self.startCameraBtn.setEnabled(False)

        self.stopDetectBtn = QtWidgets.QPushButton('🛑 停止检测')
        self.stopDetectBtn.setStyleSheet(button_style)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        self.stopDetectBtn.setEnabled(False)

        actions_layout.addWidget(self.openFileBtn)
        actions_layout.addWidget(self.startDetectBtn)
        actions_layout.addWidget(self.startCameraBtn)
        actions_layout.addWidget(self.stopDetectBtn)
        
        control_layout.addWidget(settings_group, 1)
        control_layout.addWidget(actions_group, 2)

        return control_layout

    def set_background_image(self, image_path):
        if not os.path.exists(image_path):
            return
        palette = self.palette()
        pixmap = QtGui.QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(scaled_pixmap))
        self.setPalette(palette)
        
    def toggle_segmentation(self, state):
        self.segmentation_enabled = state
        status = "启用" if self.segmentation_enabled else "禁用"
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 图像分割功能已{status}。')

    def load_model(self):
        filename = self.selectModel.currentText()
        if not filename:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 没有可加载的模型。')
            return
            
        full_path = os.path.join(self.folder_path, filename + '.pt')
        self.base_name = filename
        
        if os.path.exists(full_path):
            self.stop_detect()
            try:
                self.model = YOLO(full_path)
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model.to(self.device)
                self.sam_model = load(self.device)
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载成功: {filename} (设备: {self.device.upper()})')
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请上传文件或打开摄像头进行检测。')

                self.confidence_widget.setEnabled(True)
                self.segmentation_checkbox.setEnabled(True)
                self.openFileBtn.setEnabled(True)
                self.startCameraBtn.setEnabled(True)
            except Exception as e:
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载失败: {e}')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型文件不存在！路径: {full_path}')

    def upload_file(self):
        self.stop_detect()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请选择图片或视频文件...')
        
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择检测文件", 
            QDir.currentPath(),
            "媒体文件 (*.jpg *.jpeg *.png *.mp4 *.avi)"
        )
        
        if not file_path:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 已取消文件选择。')
            return

        self.file_path = file_path
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 已选择文件: {os.path.basename(file_path)}')
        
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png']:
            pixmap = QtGui.QPixmap(file_path)
            self.oriVideoLabel.setPixmap(pixmap.scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.startDetectBtn.setEnabled(True)
        elif file_extension in ['.mp4', '.avi']:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 无法打开视频文件！')
                self.cap = None
                self.file_path = None
                return
            ret, frame = self.cap.read()
            if ret:
                self._display_cv_frame(frame, self.oriVideoLabel)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.startDetectBtn.setEnabled(True)
        
        self.stopDetectBtn.setEnabled(True)

    def start_camera_detect(self):
        self.stop_detect()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 正在启动摄像头...')
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 无法打开摄像头！')
            self.cap = None
            return

        self.file_path = "camera_live"
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 摄像头已开启，开始实时检测...')
        
        self.timer.start(33)
        
        # --- 更新按钮状态 ---
        # 锁定模型加载和文件上传，但允许调整参数
        self.loadModel.setEnabled(False)
        self.selectModel.setEnabled(False)
        self.openFileBtn.setEnabled(False)
        self.startDetectBtn.setEnabled(False)
        self.startCameraBtn.setEnabled(False)
        self.stopDetectBtn.setEnabled(True)
        
        # <<< 关键改动：这里的参数设置控件在检测中保持可用
        # self.confidence_widget.setEnabled(False)  <-- 保持可用
        # self.segmentation_checkbox.setEnabled(False) <-- 保持可用

    def detect_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 视频源已断开！')
            self.stop_detect()
            return

        ret, frame = self.cap.read()
        if ret:
            self._display_cv_frame(frame, self.oriVideoLabel)

            results = self.model(frame, imgsz=640, conf=self.spinbox.value(), device=self.device)
            annotated_frame = results[0].plot()

            # 根据 self.segmentation_enabled 的当前值决定是否执行分割
            if self.segmentation_enabled:
                masks = inference_with_boxes(
                    frame,
                    results[0].boxes.xyxy.cpu().numpy(),
                    model=self.sam_model,
                    device=self.device
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
            
            self._display_cv_frame(final_frame, self.detectlabel)
        
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频播放/检测完成！')
            self.stop_detect()

    def _display_cv_frame(self, frame, label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_image = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_detect(self):
        if not self.file_path:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 警告: 请先上传文件！')
            return
            
        if self.model is None:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型未加载！')
            return
            
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png']:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 开始图片检测...')
            frame = cv2.imread(self.file_path)
            results = self.model(frame, imgsz=640, conf=self.spinbox.value(), device=self.device)
            annotated_frame = results[0].plot()

            if self.segmentation_enabled:
                masks = inference_with_boxes(
                    frame,
                    results[0].boxes.xyxy.cpu().numpy(),
                    model=self.sam_model,
                    device=self.device
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
                    annotated_frame = cv2.addWeighted(mask_overlay, alpha, annotated_frame, 1 - alpha, 0)

            self._display_cv_frame(annotated_frame, self.detectlabel)
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 图片检测完成！')
            
        elif file_extension in ['.mp4', '.avi']:
            if self.cap and self.cap.isOpened() and not self.timer.isActive():
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 开始视频检测...')
                self.timer.start(33)
                
                # --- 更新按钮状态 ---
                # 锁定模型加载和文件上传，但允许调整参数
                self.loadModel.setEnabled(False)
                self.selectModel.setEnabled(False)
                self.openFileBtn.setEnabled(False)
                self.startDetectBtn.setEnabled(False)
                self.startCameraBtn.setEnabled(False)
                self.stopDetectBtn.setEnabled(True)
                
                # <<< 关键改动：这里的参数设置控件在检测中保持可用
                # self.confidence_widget.setEnabled(False)  <-- 保持可用
                # self.segmentation_checkbox.setEnabled(False) <-- 保持可用

    def stop_detect(self):
        is_active = self.timer.isActive()
        if is_active:
            self.timer.stop()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.init_labels()
        if is_active or self.file_path:
             self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 检测已停止。')

        self.file_path = None

        # 重置按钮状态
        self.loadModel.setEnabled(True)
        self.selectModel.setEnabled(True)
        if self.model:
            self.confidence_widget.setEnabled(True)
            self.segmentation_checkbox.setEnabled(True)
            self.openFileBtn.setEnabled(True)
            self.startCameraBtn.setEnabled(True)
        else:
            self.confidence_widget.setEnabled(False)
            self.segmentation_checkbox.setEnabled(False)
            self.openFileBtn.setEnabled(False)
            self.startCameraBtn.setEnabled(False)

        self.startDetectBtn.setEnabled(False)
        self.stopDetectBtn.setEnabled(False)

    def init_labels(self):
        self.oriVideoLabel.clear()
        self.detectlabel.clear()
        self.oriVideoLabel.setText("原始图像")
        self.detectlabel.setText("检测结果")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())