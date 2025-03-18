# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import cv2
import threading
import base64
import time
import numpy as np
import os
import sys
import curses

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Recording,
    Record,
    Config,
    SpeedMotor,
    SteerMotor,
    Speed,
    Steer,
)
from src.templates.threadwithstop import ThreadWithStop

from src.utils.CarControl.CarControl import CarControl

class threadManualControl(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, Speed, Steer, debugger):
        super(threadManualControl, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.video_writer = ""
        self.subscribe()
        self.Configs()
        self.stdscr = curses.initscr()
        curses.cbreak()
        self.stdscr.keypad(1)
        self.speed = 0
        self.angle = 0
        self.Speed, self.Steer = Speed, Steer
        # pipeRecvIMU, pipeSendIMU = Pipe(duplex=False)
        # self.pipeRecvIMU = pipeRecvIMU
        # self.pipeSendIMU = pipeSendIMU
        # pipeRecvVLX, pipeSendVLX = Pipe(duplex=False)
        # self.pipeRecvVLX = pipeRecvVLX
        # self.pipeSendVLX = pipeSendVLX
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        print('Initialize manual control thread!!!')

    def Queue_Sending(self):
        self.control.setSpeed(self.speed)
        self.control.setAngle(self.angle)

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadManualControl", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadManualControl", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        # cv2.destroyAllWindows()
        self.speed = 0
        self.angle = 0
        self.Queue_Sending()
        super(threadManualControl, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            # print(self.control.getIMUdata())
            key = self.stdscr.getch()
            # Speed
            if  key & 0xFF == ord('q'):
                self.speed = 0
                self.Queue_Sending()
            elif key & 0xFF == ord('w'):
                # if self.speed < 50:
                #     self.speed += 1
                self.speed = 15
                self.Queue_Sending()
            elif key & 0xFF == ord('s'):
                # if self.speed > -50:
                #     self.speed -= 1
                self.speed = -30
                self.Queue_Sending()
            elif key & 0xFF == ord('z'):
                self.control.setControl(20,15,10)
            elif key & 0xFF == ord('x'):
                self.control.setControl(20,-15,10)

            
            # Steer
            if  key & 0xFF == ord('r'):
                self.angle = 0
                self.Queue_Sending()
            elif key & 0xFF == ord('d'):
                # if self.angle < 30:
                #     self.angle += 1
                self.angle = 25
                self.Queue_Sending()
            elif key & 0xFF == ord('a'):
                # if self.angle > -30:
                #     self.angle -= 1
                self.angle = -25
                self.Queue_Sending()
            # print(f'Angle: {self.angle}')
            # print(f'Speed: {self.speed}')
            elif key & 0xFF == ord('t'):
                    self.control.enVLX(1000)
            elif key & 0xFF == ord('y'):
                self.control.enIMU(200)

    # =============================== START ===============================================
    def start(self):
        super(threadManualControl, self).start()

        
