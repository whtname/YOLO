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
        self.timer1 = QtCore.QTimer()
        self.cap = None
        self.video = None
        self.file_path = None
        self.base_name = None
        self.timer1.timeout.connect(self.video_show)
 
    def init_gui(self):
        self.folder_path = CHECKPOINTS_DIR  # 自定义修改：设置模型文件夹路径
        self.setFixedSize(1600, 800)
        self.setWindowTitle('目标检测')  # 自定义修改：设置窗口名称
        self.setWindowIcon(QIcon("logo.jpg"))  # 自定义修改：设置窗口图标
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 自定义修改：设置窗口背景图
        self.set_background_image('./ui/bg.png')  

        # 界面上半部分： 视频框
        topLayout = QtWidgets.QHBoxLayout()
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.detectlabel = QtWidgets.QLabel(self)
        self.oriVideoLabel.setFixedSize(530, 400)
        self.detectlabel.setFixedSize(600, 600)
        self.oriVideoLabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top:75px;')
        self.detectlabel.setStyleSheet('border: 2px solid #ccc; border-radius: 10px; margin-top: 75px;')
        topLayout.addWidget(self.oriVideoLabel)
        topLayout.addWidget(self.detectlabel)
        main_layout.addLayout(topLayout)

        # 创建日志打印文本框
        self.outputField = QtWidgets.QTextBrowser()
        self.outputField.setFixedSize(530, 180)
        main_layout.addWidget(self.outputField)
        main_layout.setContentsMargins(150, 20, 20, 0)
        # 界面下半部分： 按钮区域
        bottomLayout = QtWidgets.QHBoxLayout()
        
        # 左侧按钮区域
        leftBtnLayout = QtWidgets.QVBoxLayout()
        
        # 模型选择下拉框
        selectModel_layout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QComboBox()
        self.selectModel.setFixedSize(200, 50)
        self.selectModel.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei";')
        
        # 遍历文件夹并添加文件名到下拉框
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.pt'):  # 确保是文件且后缀为.pt
                base_name = os.path.splitext(filename)[0]
                self.selectModel.addItem(base_name)
        
        # 添加加载模型按钮
        self.loadModel = QtWidgets.QPushButton('🔄️加载模型')  # 新建加载模型按钮
        self.loadModel.setFixedSize(100, 50)
        self.loadModel.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
            }  
            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.loadModel.clicked.connect(self.load_model) # 绑定load_model函数进行模型加载
        selectModel_layout.addWidget(self.selectModel) # 将下拉框加入到页面布局当中
        selectModel_layout.addWidget(self.loadModel)  # 将按钮加入到页面布局当中
        leftBtnLayout.addLayout(selectModel_layout)

        # 置信度阈值设置
        self.confudence_slider = QtWidgets.QWidget()
        conf_layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        
        self.con_label = QtWidgets.QLabel('置信度阈值', self)
        self.con_label.setStyleSheet('font-size: 14px; font-family: "Microsoft YaHei";')
        
        # 创建一个QSlider，范围从0到99（代表0.01到0.99）
        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)  # 0.01
        self.slider.setMaximum(99)  # 0.99
        self.slider.setValue(50)  # 0.5
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setFixedSize(170, 30)
        
        # 创建一个QDoubleSpinBox用于显示和设置滑动条的值
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
        
        # 连接信号和槽
        self.slider.valueChanged.connect(self.updateSpinBox)
        self.spinbox.valueChanged.connect(self.updateSlider)
        
        leftBtnLayout.addWidget(self.confudence_slider)

        # 文件上传按钮
        self.openImageBtn = QtWidgets.QPushButton('🖼️文件上传')
        self.openImageBtn.setFixedSize(100, 65)
        self.openImageBtn.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px; 
                margin-bottom: 15px;
            }  
            
            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.openImageBtn.clicked.connect(self.upload_file) # 绑定upload_file事件
        self.openImageBtn.setEnabled(False) # 初始化按钮默认不可操作，加载模型之后可以操作
        leftBtnLayout.addWidget(self.openImageBtn)

        # 右侧按钮区域
        rightBtnLayout = QtWidgets.QVBoxLayout()
        
        # 执行预测按钮
        self.start_detect = QtWidgets.QPushButton('🔍开始检测')
        self.start_detect.setFixedSize(100, 50)
        self.start_detect.setStyleSheet("""
            QPushButton {  
                background-color: white; /* 正常状态下的背景颜色 */  
                border: 2px solid gray;  /* 正常状态下的边框 */  
                border-radius: 10px;  
                padding: 5px;
                font-size: 14px;
            }  
            
            QPushButton:hover {  
                background-color: #f0f0f0;  /* 悬停状态下的背景颜色 */  
            }  
        """)
        self.start_detect.clicked.connect(self.show_detect) # 绑定show_detect函数事件
        self.start_detect.setEnabled(False)
        rightBtnLayout.addWidget(self.start_detect)

        # 停止检测按钮
        self.stopDetectBtn = QtWidgets.QPushButton('🛑停止')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect) # 绑定stop_detect中断检测事件
        rightBtnLayout.addWidget(self.stopDetectBtn)

        bottomLayout.addLayout(leftBtnLayout)
        bottomLayout.addLayout(rightBtnLayout)
        main_layout.addLayout(bottomLayout)

    def set_background_image(self, image_path):
        palette = self.palette()
        if os.path.exists(image_path):
            palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(QtGui.QPixmap(image_path)))
            self.setPalette(palette)

    def load_model(self):
        filename = self.selectModel.currentText()
        full_path = os.path.join(self.folder_path, filename + '.pt')
        self.base_name = os.path.splitext(os.path.basename(full_path))[0]
        if full_path.endswith('.pt'):
            # 加载预训练模型
            self.model = YOLO(full_path)
            self.start_detect.setEnabled(True)
            self.stopDetectBtn.setEnabled(True)
            self.openImageBtn.setEnabled(True)
            self.confudence_slider.setEnabled(True)
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 模型加载成功: {filename}')
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请选择置信度阈值')
        else:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请重新选择模型文件！')
            print("Reselect model")

    def updateSpinBox(self, value):
        self.spinbox.setValue(value / 100)

    def updateSlider(self, value):
        self.slider.setValue(int(value * 100))

    def upload_file(self):
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请选择检测文件')
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDirectory(QDir("./valid_file"))  # 修改上传文件路径
        # 对上传的文件根据后缀名称进行过滤
        file_path, file_type = file_dialog.getOpenFileName(self, "选择检测文件", filter='*.jpg *.mp4')
        
        if file_path:
            self.file_path = file_path
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 已选择文件: {os.path.basename(file_path)}')
            
            if file_path.endswith('.jpg'):
                # 显示原始图片
                pixmap = QtGui.QPixmap(file_path)
                self.oriVideoLabel.setPixmap(pixmap.scaled(self.oriVideoLabel.size(), Qt.KeepAspectRatio))
            elif file_path.endswith('.mp4'):
                self.cap = cv2.VideoCapture(file_path)
                self.timer1.start(30)  # 30ms更新一帧

    def video_show(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(
                    self.oriVideoLabel.size(), Qt.KeepAspectRatio))
            else:
                self.cap.release()
                self.timer1.stop()

    def show_detect(self):
        if not self.file_path:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 请先选择检测文件！')
            return
            
        self.value = self.spinbox.value()
        
        if self.file_path.endswith('.jpg'):
            # 图片检测
            frame = cv2.imread(self.file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.model is None:
                self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型未加载！')
                return
            # 执行检测
            results = self.model(frame, imgsz=[448, 352], 
                               device='cuda' if torch.cuda.is_available() else 'cpu', 
                               conf=self.value)
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            
            # 显示结果
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(annotated_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(
                self.detectlabel.size(), Qt.KeepAspectRatio))
            
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 图片检测完成！')
            
        elif self.file_path.endswith('.mp4'):
            # 视频检测
            if not self.cap:
                self.cap = cv2.VideoCapture(self.file_path)
            
            self.timer.timeout.connect(self.detect_frame)
            self.timer.start(30)
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 开始视频检测...')

    def detect_frame(self):
        if self.cap is None:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 视频未加载！')
            self.timer.stop()
            return
        ret, frame = self.cap.read()
        if self.model is None:
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 错误: 模型未加载！')
            return
        if ret:
            # 执行检测
            results = self.model(frame, imgsz=[448, 352], 
                              device='cuda' if torch.cuda.is_available() else 'cpu', 
                              conf=self.value)
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # 显示原始帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(
                self.oriVideoLabel.size(), Qt.KeepAspectRatio))
            
            # 显示检测结果
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(annotated_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(
                self.detectlabel.size(), Qt.KeepAspectRatio))
        else:
            self.timer.stop()
            self.cap.release()
            self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 视频检测完成！')

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.timer1.isActive():
            self.timer1.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video = None
        self.ini_labels()
        self.outputField.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - 检测中断！')
        self.file_path = None

    def ini_labels(self):
        self.oriVideoLabel.clear()
        self.detectlabel.clear()
        self.oriVideoLabel.setText("原始视频")
        self.detectlabel.setText("检测结果")