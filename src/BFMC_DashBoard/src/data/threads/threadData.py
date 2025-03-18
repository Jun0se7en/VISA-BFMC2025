import cv2
import threading
import socket
import base64
import time
import numpy as np
import os
import io
import csv
import json
import random

from multiprocessing import Pipe
from src.templates.threadwithstop import ThreadWithStop
import struct
import pickle
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsColorizeEffect
from PyQt5.QtGui import QFontDatabase, QPixmap, QImage
from PyQt5.QtCore import QThread, QEvent, pyqtSignal

import folium
from ui_interface import *


class threadData(QThread):
    DataUpdate = pyqtSignal(int, int, list, list)

    def __init__(self, ip_address, port):
        super().__init__()
        self.ThreadActive = True
        self.PORT = port
        self.SERVER_ADDRESS = ip_address
        self.connect_flag = False
        self.prev_state = ""
        self.state = ""
        self.fps = 0
        self.speed = 0
        self.angle = 0

    def run(self):
        self.ThreadActive = True
        self.client_socket = socket.socket()  # instantiate
        print(f'Socker with {self.PORT} created')
        while not self.connect_flag:
            try:
                self.client_socket.connect((self.SERVER_ADDRESS, self.PORT))  # connect to the server
                self.connect_flag = True
            except:
                print('Connecting Failed!!! Retrying....')
                pass

        data = b""
        payload_size = struct.calcsize("Q")
        while self.ThreadActive:
            while len(data) < payload_size:
                packet = self.client_socket.recv(4*1024)
                if not packet: break
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q",packed_msg_size)[0]
            
            while len(data) < msg_size:
                data += self.client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data  = data[msg_size:]
            frame = pickle.loads(frame_data)
            # print("Data:",frame)
            position = frame["Position"]

            x, y = 0, 0
            if position is not None:
                x = position["x"]
                y = position["y"]

            frame["Stats"][-3] = int(frame["Stats"][-3]*100)
            stats = [list(map(int,frame["Stats"][:len(frame["Stats"])-2])), int(frame["Stats"][-1])]

            if frame["CarStats"] is not None:
                self.fps = frame["CarStats"][0]
                self.state = frame["CarStats"][1]
                self.speed = int(frame["CarStats"][2]/2)
                self.angle = int(frame["CarStats"][3])
            
            label = [x, y, self.state, self.prev_state, int(frame["Stats"][-2]/1000), int(self.fps)]

            self.DataUpdate.emit(self.speed, self.angle, label, stats)

            if self.prev_state != self.state:
                self.prev_state = self.state


            # phát tín hiệu về Main Thread
            
        # When everything done, release the socket
        self.client_socket.close()
        self.terminate()

    def stop(self):
        self.ThreadActive = False
        self.terminate()
