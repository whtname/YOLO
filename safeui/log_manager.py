# src/managers/log_manager.py

from PySide6.QtWidgets import QPlainTextEdit, QScrollBar
from PySide6.QtCore import Qt, Slot, QDateTime

class LogManager:
    # 定义日志级别颜色映射 (HTML 颜色代码)
    # 针对 #343434 背景色进行优化
    LOG_COLORS = {
        "INFO": "#00c4f4",    # 默认信息，浅灰色，清晰
        "WARNING": "#FFDD44", # 警告，亮黄色，非常显眼
        "ERROR": "#FF6666",   # 错误，亮红色，强烈警示
        "DEBUG": "#99BBFF",   # 调试信息，柔和的蓝色
        "CRITICAL": "#FF3333" # 严重错误，更深的亮红色
    }

    def __init__(self, main_window_instance):
        self.main_window = main_window_instance
        self.ui = main_window_instance.ui

        self.log_output_text_edit = self.ui.logOutputTextEdit

        self._setup_log_area()

    def _setup_log_area(self):
        self.log_output_text_edit.setReadOnly(True)
        self.log_output_text_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth) # 允许换行

        self.log_output_text_edit.setFixedHeight(70) # 你可以根据需要调整这个值

        self.log_output_text_edit.setObjectName("logOutputTextEdit")

    def _format_log_message(self, level: str, source: str, message: str) -> str:
        """
        格式化日志消息为带颜色的 HTML 字符串，整行统一颜色。
        :param level: 日志级别 (e.g., "INFO", "WARNING", "ERROR")
        :param source: 消息来源 (e.g., "MainApp", "ModuleA", "Network")
        :param message: 具体日志内容
        :return: 格式化后的 HTML 字符串
        """
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz")
        color = self.LOG_COLORS.get(level.upper(), self.LOG_COLORS["INFO"]) # 默认为 INFO 颜色

        # 使用一个 <span> 标签包裹整行内容，统一设置颜色
        formatted_message = (
            f"<span style='color:{color};'>"
            f"[{timestamp}] "
            f"[{level.upper()}] "
            f"[{source}] "
            f"{message}" # 消息内容直接包含在同一个 span 内
            f"</span>"
        )
        return formatted_message

    @Slot(str, str, str)
    def append_log_message(self, level: str, source: str, message: str):
        """
        在日志输出框中追加格式化后的彩色 HTML 文本。
        :param level: 日志级别 (e.g., "INFO", "WARNING", "ERROR")
        :param source: 消息来源
        :param message: 日志内容
        """
        html_message = self._format_log_message(level, source, message)
        self.log_output_text_edit.appendHtml(html_message)
        # 自动滚动到最新日志 (底部)
        self.log_output_text_edit.verticalScrollBar().setValue(self.log_output_text_edit.verticalScrollBar().maximum())