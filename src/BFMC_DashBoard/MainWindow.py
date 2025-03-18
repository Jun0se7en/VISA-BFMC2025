import os
import sys
import io
import traceback
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsColorizeEffect
from PyQt5.QtGui import QFontDatabase, QPixmap, QImage, QPen, QPainter
from PyQt5.QtCore import QThread, QEvent, pyqtSignal, QPointF, Qt
import pyqtgraph as pg
import folium
import cv2
import socket
import pickle
import struct
import csv
from time import sleep
from threading import Thread
from multiprocessing.sharedctypes import Value
import signal
import numpy as np
from ui_interface import *
from src.client.threads.threadClient import threadClient
from src.data.threads.threadData import threadData

class MainWindow(QMainWindow):     
    def __init__(self, Camera, Data, xavierip, xavierport, width, height, parent=None):
        print("MainWindow.__init__ started")
        QMainWindow.__init__(self)
        
        self.birdview = Value('i', 0)
        self.calibrate = Value('i', 0)
        self.Camera = Camera
        self.Data = Data
        self.xavierip = xavierip
        self.xavierport = xavierport
        self.allProcesses = []

        try:
            print("Setting up UI...")
            self.ui = Ui_MainWindow(width, height)
            print("Ui_MainWindow created")
            self.ui.setupUi(self)
            print("UI setup complete")
            print(f"Window size: {self.width()}x{self.height()}")
            print(f"Central widget size: {self.ui.centralwidget.width()}x{self.ui.centralwidget.height()}")

            # Load background image
            background_image_path = os.path.abspath("BackGround.png")
            print(f"Attempting to load background from: {background_image_path}")
            if not os.path.exists(background_image_path):
                print(f"Background image not found: {background_image_path}")
                self.background_pixmap = None
            else:
                self.background_pixmap = QPixmap(background_image_path)
                if self.background_pixmap.isNull():
                    print("Failed to load background pixmap - image may be corrupt or invalid")
                else:
                    print(f"Background pixmap loaded successfully: {self.background_pixmap.width()}x{self.background_pixmap.height()}")

            # Ensure child widgets are transparent
            self.ui.Header_Frame.setStyleSheet("background-color: rgba(0, 0, 0, 0); border: none;")
            self.ui.BodyFrame.setStyleSheet("background-color: rgba(0, 0, 0, 0); border: none;")
            # self.ui.WebviewFrame.setStyleSheet("background-color: rgba(0, 0, 0, 0); border: none;")
            self.ui.lb_Raw_Img.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
            self.ui.lb_Output_Img.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
            # self.ui.WebView.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

            print("Showing window...")
            self.show()

            # HIDE LEFT/RIGHT ARROW SIGNAL
            self.ui.lb_Left_Signal.setVisible(False)
            self.ui.lb_Right_Signal.setVisible(False)

            # # CUSTOMIZE ANALOGUE GAUGE SPEED WIDGET
            # self.ui.Analog_Gauge_Speed.enableBarGraph = True
            # self.ui.Analog_Gauge_Speed.valueNeedleSnapzone = 1

            # # Set Angle Offset
            # Speed_Gauge_Offset = 0
            # self.ui.Analog_Gauge_Speed.updateAngleOffset(Speed_Gauge_Offset)

            # # Set gauge units
            # self.ui.Analog_Gauge_Speed.units = "Km/h"

            # # Set minimum/maximum gauge value
            # self.ui.Analog_Gauge_Speed.minValue = -50
            # self.ui.Analog_Gauge_Speed.maxValue = 50

            # # Set scale divisions
            # self.ui.Analog_Gauge_Speed.scalaCount = 10
            # self.ui.Analog_Gauge_Speed.updateValue(int(self.ui.Analog_Gauge_Speed.maxValue - self.ui.Analog_Gauge_Speed.minValue)/2)

            # # Select gauge theme
            # self.ui.Analog_Gauge_Speed.setCustomGaugeTheme(color1="red", color2="orange", color3="green")
            # self.ui.Analog_Gauge_Speed.setNeedleCenterColor(color1="dark gray")
            # self.ui.Analog_Gauge_Speed.setOuterCircleColor(color1="dark gray")
            # self.ui.Analog_Gauge_Speed.setBigScaleColor("yellow")
            # self.ui.Analog_Gauge_Speed.setFineScaleColor("blue")

            # # CUSTOMIZE ANALOGUE GAUGE ANGLE WIDGET
            # self.ui.Analog_Gauge_Angle.enableBarGraph = True
            # self.ui.Analog_Gauge_Angle.valueNeedleSnapzone = 1

            # # Set Angle Offset
            # Angle_Gauge_Offset = 0
            # self.ui.Analog_Gauge_Angle.updateAngleOffset(Angle_Gauge_Offset)

            # # Set gauge units
            # self.ui.Analog_Gauge_Angle.units = "Degree"

            # # Set minimum/maximum gauge value
            # self.ui.Analog_Gauge_Angle.minValue = -50
            # self.ui.Analog_Gauge_Angle.maxValue = 50

            # # Set scale divisions
            # self.ui.Analog_Gauge_Angle.scalaCount = 10
            # self.ui.Analog_Gauge_Angle.updateValue(int(self.ui.Analog_Gauge_Speed.maxValue - self.ui.Analog_Gauge_Speed.minValue)/2)

            # # Select gauge theme
            # self.ui.Analog_Gauge_Angle.setCustomGaugeTheme(color1="red", color2="orange", color3="green")
            # self.ui.Analog_Gauge_Angle.setNeedleCenterColor(color1="dark gray")
            # self.ui.Analog_Gauge_Angle.setOuterCircleColor(color1="dark gray")
            # self.ui.Analog_Gauge_Angle.setBigScaleColor("yellow")
            # self.ui.Analog_Gauge_Angle.setFineScaleColor("blue")

            if self.Camera:
                self.CameraWorker = threadClient(self.xavierip, self.xavierport, self.birdview, False, self.calibrate)
                self.CameraWorker.start()
                self.CameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

            if self.Camera:
                self.OutputImageWorker = threadClient(self.xavierip, self.xavierport+1, self.birdview, True, self.calibrate)
                self.OutputImageWorker.start()
                self.OutputImageWorker.ImageUpdate.connect(self.OutputImageUpdateSlot)

            if self.Data:
                self.DataWorker = threadData(self.xavierip, self.xavierport+2)
                self.DataWorker.start()
                self.DataWorker.DataUpdate.connect(self.UpdateData)

            signal.signal(signal.SIGINT, self.signal_handler)

        except Exception as e:
            print(f"Error in MainWindow.__init__: {str(e)}")
            print(traceback.format_exc())
            raise

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.background_pixmap and not self.background_pixmap.isNull():
            # Draw on centralwidget’s rectangle
            painter.drawPixmap(self.ui.centralwidget.rect(), self.background_pixmap)
            # print("Background drawn on centralwidget")
        else:
            print("No valid background pixmap to draw")
        super().paintEvent(event)

    def signal_handler(self, signal, frame):
        print("\nCatching a Keyboard Interruption exception! Shutdown all processes.\n")
        if self.Camera:
            self.CameraWorker.stop()
            self.OutputImageWorker.stop()
        if self.Data:
            self.DataWorker.stop()
        sys.exit(0)

    def ImageUpdateSlot(self, image):
        self.ui.lb_Raw_Img.setPixmap(QPixmap.fromImage(image))

    def OutputImageUpdateSlot(self, image):
        self.ui.lb_Output_Img.setPixmap(QPixmap.fromImage(image))

    def UpdateData(self, speed, angle, label, data):
        ### Speed ###
        speed = min(60, speed)
        self.ui.update_speed(speed)

        ### Angle ###
        if angle < 0:
            angle = max(-25, angle)
        else:
            angle = min(25, angle)
        self.ui.update_angle(angle)

        x, y, state, prev_state, cur_pow, fps = label
        ### Update Label ###
        # Example usage
        self.ui.update_label(self.ui.X_Value, str(x))
        self.ui.update_label(self.ui.Y_Value, str(y))
        self.ui.update_label(self.ui.Current_State, state)
        self.ui.update_label(self.ui.Prev_State, prev_state)
        self.ui.update_label(self.ui.Current_Power, cur_pow)
        self.ui.update_label(self.ui.FPS, str(fps))

        ### Update Plot ###
        self.ui.update_plot([x], [y])

        ### Update Progress ###
        progress, temp = data
        self.ui.update_progress(progress, temp)

        

    def keyPressEvent(self, event):
        try:
            key = chr(event.key()).lower()
            if key == 'b':
                print(f'Birdview: {bool(self.birdview.value)}')
                self.birdview.value = not self.birdview.value
            elif key == 'c':
                print(f'Calibrate: {bool(self.calibrate.value)}')
                self.calibrate.value = not self.calibrate.value
            else:
                key = ord(key)
                self.CameraWorker.send_key(key)
        except:
            pass
        
    def keyReleaseEvent(self, event):
        pass