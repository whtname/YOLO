#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :app.py.py
# @Time      :2025/7/8 15:36:12
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap # 用于处理图像

from testui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.upload_image.setText("上传图片")
        self.ui.upload_image.clicked.connect(self.upload_images)

    def upload_images(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                            "选择图片",
                                "",
                                "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if  file_path:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.ui.result_iamge.size(), Qt.KeepAspectRatio)
                self.ui.result_iamge.setPixmap(scaled_pixmap)
            else:
                print("无法加载图片")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
