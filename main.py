from PySide6 import QtWidgets
from ui.ui import MyWindow 
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MyWindow()
    window.show()
    app.exec()