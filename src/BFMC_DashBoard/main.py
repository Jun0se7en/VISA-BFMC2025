import sys
import cv2
sys.path.append(".")
from multiprocessing import Queue, Event
import multiprocessing
import logging
import argparse
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsColorizeEffect
from PyQt5.QtGui import QFontDatabase, QPixmap, QImage
from PyQt5.QtCore import QThread, QEvent, pyqtSignal

import folium
from MainWindow import MainWindow


# ===================================== PROCESS IMPORTS ==================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Process Input IP')
    parser.add_argument('--xavierip', type=str, help='Xavier IP', default="192.168.7.247")
    parser.add_argument('--xavierport', type=int, help='Xavier Port', default=12345)
    parser.add_argument('--width', type=int, help='Width', default=1280)
    parser.add_argument('--height', type=int, help='Height', default=720)
    args = parser.parse_args()
    app = QApplication(sys.argv)

    Camera = True
    Data = True
    window = MainWindow(Camera, Data, args.xavierip, args.xavierport, args.width, args.height)
    window.show()

    # ===================================== STAYING ALIVE ====================================
    sys.exit(app.exec_())
    