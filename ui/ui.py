import os
from datetime import datetime
import cv2
import torch
from PyQt5.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QDir
from ultralytics import YOLO
from utils.paths import CHECKPOINTS_DIR

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
 
        self.init_gui()
        self.model = None
        self.timer = QtCore.QTimer() 
        self.cap = None
        self.video = None 
        self.file_path = None
        self.base_name = None
        self.timer.timeout.connect(self.detect_frame)
 
    def init_gui(self):
        self.folder_path = CHECKPOINTS_DIR
        self.setFixedSize(1400, 800)
        self.setWindowTitle('目标检测')
        self.setWindowIcon(QIcon("logo.jpg"))
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20) # 设置整体布局的边距
        self.set_background_image('./ui/bg.png')  

        # 创建一个左侧的垂直布局，用于放置原始视频和输出日志
        left_display_layout = QtWidgets.QVBoxLayout()
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.oriVideoLabel.setFixedSize(530, 400)
        # 调整 margin-top 使图像框上移
        self.oriVideoLabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top:20px;')
        self.oriVideoLabel.setAlignment(Qt.AlignCenter)
        self.oriVideoLabel.setText("原始视频")
        left_display_layout.addWidget(self.oriVideoLabel)

        self.outputField = QtWidgets.QTextBrowser()
        self.outputField.setFixedSize(530, 180)
        left_display_layout.addWidget(self.outputField)
        left_display_layout.setContentsMargins(0, 0, 0, 0) # 移除内部布局的额外边距
        left_display_layout.addStretch(1) # 使内容向上对齐

        # 创建右侧的检测结果标签
        self.detectlabel = QtWidgets.QLabel(self)
        self.detectlabel.setFixedSize(600, 600)
        # 调整 margin-top 使图像框上移
        self.detectlabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top: 20px;')
        self.detectlabel.setAlignment(Qt.AlignCenter)
        self.detectlabel.setText("检测结果")

        # 将左侧显示布局和右侧检测结果标签放入一个水平布局
        top_area_layout = QtWidgets.QHBoxLayout()
        top_area_layout.addLayout(left_display_layout)
        top_area_layout.addWidget(self.detectlabel)
        top_area_layout.setContentsMargins(150, 20, 20, 0) # 整体顶部区域的边距

        main_layout.addLayout(top_area_layout)
        # 在图像框区域和底部按钮区域之间添加一个固定间距
        main_layout.addSpacing(20) # 调整此值以控制间距大小

        bottomLayout = QtWidgets.QHBoxLayout()
        
        leftBtnLayout = QtWidgets.QVBoxLayout()
        
        selectModel_layout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QComboBox()
        self.selectModel.setFixedSize(200, 50)
        self.selectModel.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei";')
        
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):
                base_name = os.path.splitext(filename)[0]
                self.selectModel.addItem(base_name)
        
        self.loadModel = QtWidgets.QPushButton('🔄️加载模型')
        self.loadModel.setFixedSize(100, 50)
        self.loadModel.setStyleSheet("""
            QPushButton {  
                background-color: white;
                border: 2px solid gray;  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
            }  
            QPushButton:hover {  
                background-color: #f0f0f0;  
            }  
        """)
        self.loadModel.clicked.connect(self.load_model)
        selectModel_layout.addWidget(self.selectModel)
        selectModel_layout.addWidget(self.loadModel)
        leftBtnLayout.addLayout(selectModel_layout)

        self.confudence_slider = QtWidgets.QWidget()
        conf_layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        
        self.con_label = QtWidgets.QLabel('置信度阈值', self)
        self.con_label.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei";')
        
        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(99)
        self.slider.setValue(50)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setFixedSize(170, 30)
        
        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinbox.setMinimum(0.01)
        self.spinbox.setMaximum(0.99)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setValue(0.5)
        self.spinbox.setDecimals(2)
        self.spinbox.setFixedSize(60, 30)
        self.spinbox.setStyleSheet('border: 2px solid gray; border-radius: 10px; '
                                'padding: 5px; background-color: #f0f0f0; font-size: 14px;')
        
        self.confudence_slider.setFixedSize(250, 64)
        conf_layout.addWidget(self.con_label)
        hlayout.addWidget(self.slider)
        hlayout.addWidget(self.spinbox)
        conf_layout.addLayout(hlayout)
        self.confudence_slider.setLayout(conf_layout)
        self.confudence_slider.setEnabled(False)
        
        self.slider.valueChanged.connect(self.updateSpinBox)
        self.spinbox.valueChanged.connect(self.updateSlider)
        
        leftBtnLayout.addWidget(self.confudence_slider)

        self.openImageBtn = QtWidgets.QPushButton('🖼️文件上传')
        self.openImageBtn.setFixedSize(100, 65)
        self.openImageBtn.setStyleSheet("""
            QPushButton {  
                background-color: white;
                border: 2px solid gray;  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
                margin-bottom: 15px;
            }  
            
            QPushButton:hover {  
                background-color: #f0f0f0;  
            }  
        """)
        self.openImageBtn.clicked.connect(self.upload_file)
        self.openImageBtn.setEnabled(False)
        leftBtnLayout.addWidget(self.openImageBtn)

        rightBtnLayout = QtWidgets.QVBoxLayout()
        
        self.start_detect = QtWidgets.QPushButton('🔍开始检测')
        self.start_detect.setFixedSize(100, 50)
        self.start_detect.setStyleSheet("""
            QPushButton {  
                background-color: white;
                border: 2px solid gray;  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px;
            }  
            
            QPushButton:hover {  
                background-color: #f0f0f0;  
            }  
        """)
        self.start_detect.clicked.connect(self.show_detect)
        self.start_detect.setEnabled(False)
        rightBtnLayout.addWidget(self.start_detect)

        self.startCameraBtn = QtWidgets.QPushButton('📹开始摄像头')
        self.startCameraBtn.setFixedSize(100, 50)
        self.startCameraBtn.setStyleSheet("""
            QPushButton {  
                background-color: white;
                border: 2px solid gray;  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px;
            }  
            QPushButton:hover {  
                background-color: #f0f0f0;  
            }  
        """)
        self.startCameraBtn.clicked.connect(self.start_camera_detect)
        self.startCameraBtn.setEnabled(False)
        rightBtnLayout.addWidget(self.startCameraBtn)

        self.stopDetectBtn = QtWidgets.QPushButton('🛑停止')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        rightBtnLayout.addWidget(self.stopDetectBtn)

        bottomLayout.addLayout(leftBtnLayout)
        bottomLayout.addLayout(rightBtnLayout)
        main_layout.addLayout(bottomLayout)
        main_layout.addStretch(1) # 将按钮部分向上推，以平衡布局底部空间

    def set_background_image(self, image_path):
            palette = self.palette()
            if os.path.exists(image_path):
                original_pixmap = QtGui.QPixmap(image_path)
                scaled_pixmap = original_pixmap.scaled(
                    self.size(),  # 获取当前窗口的尺寸
                    Qt.IgnoreAspectRatio, # 忽略图片原有的宽高比，完全填充窗口
                    Qt.SmoothTransformation # 使用平滑转换以获得更好的缩放质量
                )
                palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(scaled_pixmap))
                self.setPalette(palette)

    def load_model(self):
        filename = self.selectModel.currentText()
        full_path = os.path.join(self.folder_path, filename + '.pt')
        self.base_name = os.path.splitext(os.path.basename(full_path))[0]
        if full_path.endswith('.pt'):
            self.stop_detect() 
            self.model = YOLO(full_path)
            self.start_detect.setEnabled(True)
            self.openImageBtn.setEnabled(True)
            self.confudence_slider.setEnabled(True)
            self.startCameraBtn.setEnabled(True)
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载成功: {filename}')
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请选择置信度阈值')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请重新选择模型文件！')

    def updateSpinBox(self, value):
        self.spinbox.setValue(value / 100)

    def updateSlider(self, value):
        self.slider.setValue(int(value * 100))

    def upload_file(self):
        self.stop_detect()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请选择检测文件')
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDirectory(QDir("./valid_file"))
        file_path, file_type = file_dialog.getOpenFileName(self, "选择检测文件", filter='*.jpg *.mp4')
        
        if file_path:
            self.file_path = file_path
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 已选择文件: {os.path.basename(file_path)}')
            
            if file_path.endswith('.jpg'):
                pixmap = QtGui.QPixmap(file_path)
                self.oriVideoLabel.setPixmap(pixmap.scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio))
                self.start_detect.setEnabled(True)
                self.startCameraBtn.setEnabled(False)
                self.openImageBtn.setEnabled(True)
            elif file_path.endswith('.mp4'):
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 无法打开视频文件！')
                    self.cap = None
                    self.file_path = None
                    return
                self.start_detect.setEnabled(True)
                self.startCameraBtn.setEnabled(False)
                self.openImageBtn.setEnabled(True)
            self.stopDetectBtn.setEnabled(True)
        else:
            if self.model is not None:
                self.start_detect.setEnabled(True)
                self.openImageBtn.setEnabled(True)
                self.startCameraBtn.setEnabled(True)
            self.stopDetectBtn.setEnabled(False)

    def start_camera_detect(self):
        self.stop_detect()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 启动摄像头...')
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 无法打开摄像头！请检查摄像头连接或权限。')
            self.cap = None
            return

        self.file_path = "camera_live"
        self.value = self.spinbox.value()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 开始摄像头实时检测...')
        
        self.timer.start(30)
        
        self.startCameraBtn.setEnabled(False)
        self.openImageBtn.setEnabled(False)
        self.start_detect.setEnabled(False)
        self.stopDetectBtn.setEnabled(True)

    def detect_frame(self):
        if self.cap is None:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 视频/摄像头未加载或已断开！')
            self.timer.stop()
            self.stop_detect()
            return
        if self.model is None:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型未加载！')
            self.timer.stop()
            self.stop_detect()
            return

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image_ori = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(q_image_ori).scaled(
                self.oriVideoLabel.size(), Qt.KeepAspectRatio))

            results = self.model(frame, imgsz=[448, 352], 
                               device='cuda' if torch.cuda.is_available() else 'cpu', 
                               conf=self.value)
            
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            h, w, ch = annotated_frame_rgb.shape
            bytes_per_line = ch * w
            q_image_det = QtGui.QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(q_image_det).scaled(
                self.detectlabel.size(), Qt.KeepAspectRatio))
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频/摄像头流结束！')
            self.timer.stop()
            self.stop_detect()

    def show_detect(self):
        if not self.file_path:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请先选择检测文件！')
            return
            
        if self.model is None:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型未加载！')
            return
            
        self.value = self.spinbox.value()
        
        if self.file_path.endswith('.jpg'):
            frame = cv2.imread(self.file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame, imgsz=[448, 352], 
                               device='cuda' if torch.cuda.is_available() else 'cpu', 
                               conf=self.value)
            
            annotated_frame = results[0].plot()
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(annotated_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(
                self.detectlabel.size(), Qt.KeepAspectRatio))
            
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 图片检测完成！')
            self.stopDetectBtn.setEnabled(True) 
            
        elif self.file_path.endswith('.mp4'):
            if self.cap and self.cap.isOpened() and not self.timer.isActive():
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 开始视频检测...')
                self.timer.start(30)
                self.start_detect.setEnabled(False)
                self.openImageBtn.setEnabled(False)
                self.startCameraBtn.setEnabled(False)
                self.stopDetectBtn.setEnabled(True)
            elif not self.cap or not self.cap.isOpened():
                 self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 视频未加载或无法打开！')
            elif self.timer.isActive():
                 self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频检测已在进行中！')

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video = None
        self.init_labels()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 检测中断！')
        self.file_path = None

        if self.model is not None:
            self.start_detect.setEnabled(True)
            self.openImageBtn.setEnabled(True)
            self.startCameraBtn.setEnabled(True)
        self.stopDetectBtn.setEnabled(False)

    def init_labels(self):
        self.oriVideoLabel.clear()
        self.detectlabel.clear()
        self.oriVideoLabel.setText("原始视频")
        self.detectlabel.setText("检测结果") # 确保检测结果标签也重置文本