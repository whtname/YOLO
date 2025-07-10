# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'testui.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(783, 565)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(160, 20, 404, 462))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalFrame_2 = QFrame(self.verticalLayoutWidget)
        self.verticalFrame_2.setObjectName(u"verticalFrame_2")
        self.verticalFrame_2.setStyleSheet(u"border:1px solid red;\n"
"border-radius:3px;\n"
"qproperty-alignment:'AlignCenter';\n"
"font-size:14pt;\n"
"font-weight:500;")
        self.verticalLayout_2 = QVBoxLayout(self.verticalFrame_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalFrame_4 = QFrame(self.verticalFrame_2)
        self.verticalFrame_4.setObjectName(u"verticalFrame_4")
        self.verticalLayout_5 = QVBoxLayout(self.verticalFrame_4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.result_iamge = QLabel(self.verticalFrame_4)
        self.result_iamge.setObjectName(u"result_iamge")

        self.verticalLayout_5.addWidget(self.result_iamge)


        self.verticalLayout_3.addWidget(self.verticalFrame_4)


        self.verticalLayout_2.addLayout(self.verticalLayout_3)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalFrame_5 = QFrame(self.verticalFrame_2)
        self.verticalFrame_5.setObjectName(u"verticalFrame_5")
        self.verticalLayout_6 = QVBoxLayout(self.verticalFrame_5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.upload_image = QPushButton(self.verticalFrame_5)
        self.upload_image.setObjectName(u"upload_image")

        self.verticalLayout_6.addWidget(self.upload_image)


        self.verticalLayout_4.addWidget(self.verticalFrame_5)


        self.verticalLayout_2.addLayout(self.verticalLayout_4)

        self.verticalLayout_2.setStretch(0, 5)
        self.verticalLayout_2.setStretch(1, 1)

        self.verticalLayout.addWidget(self.verticalFrame_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.result_iamge.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.upload_image.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
    # retranslateUi

