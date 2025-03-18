import sys

sys.path.append(".")
import logging
import os
import time
from multiprocessing import Event, Queue
from multiprocessing.sharedctypes import Value

import torch
import argparse

from src.clearBuffer.processClearBuffer import processClearBuffer

# ===================================== PROCESS IMPORTS ==================================
from src.gateway.processGateway import processGateway
from src.hardware.camera.processCamera import processCamera
from src.imageProcessing.processBoschNet import processBoschNet
from src.server.processServer import processServer
# from src.decisionMaking.processDecisionMaking import processDecisionMaking
from src.control.manualControl.processManualControl import processManualControl
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.data.TrafficCommunication.processTrafficCommunication import processTrafficCommunication

# ======================================== SETTING UP ====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Input IP')
    parser.add_argument('--xavierip', type=str, help='Xavier IP', default="192.168.7.247")
    parser.add_argument('--xavierport', type=int, help='Xavier Port', default=12345)
    parser.add_argument('--serverip', type=str, help='Localization Server IP', default="192.168.1.111")
    parser.add_argument('--serverport', type=int, help='Localization Server Port', default=9000)
    args = parser.parse_args()

    allProcesses = list()
    queueList = {
        'Control': Queue(),
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
        # After Processed
        "ObjectDetection": Queue(),
        "Segmentation": Queue(),
        "Points": Queue(),
        "DecisionMaking": Queue(),

        # Camera
        "MainCamera": Queue(),
        "BoschNetCamera": Queue(),

        # Localization
        "Position": Queue(),

        "CarStats": Queue(),
    }
    
    Speed = Value("i", 0)
    Steer = Value("i", 0)

    logging = logging.getLogger()

    # Camera
    Camera = True
    
    # Image Processing
    BoschNet = True

    # Server
    Server = True

    # Clear Buffer (Optional Remove Further)
    ClearBuffer = True

    # Manual Control
    ManualControl = False

    # DecisionMaking
    DecisionMaking = False
    
    # Serial Handler
    SerialHandler = True
    
    # Traffic Communication
    TrafficCommunication = True

    # =========================== CHECKING NECESSARY PROCESSES ===============================
    # if not Camera:
    #     raise Exception("Camera is not initialized!!!")

    # if (ManualControl or DecisionMaking) and not SerialHandler:
    #     raise Exception("Serial Handler is not initialized!!!")

    # if not ClearBuffer:
    #     raise Exception("Clear Buffer is not initialized!!!")

    # ===================================== SETUP PROCESSES ==================================

    # Initializing gateway
    processGateway = processGateway(queueList, logging)
    allProcesses.append(processGateway)

    # Initializing camera
    if Camera:
        width = 640.0
        height = 480.0
        fps = 30
        processCamera = processCamera(queueList, logging, width, height, fps, debugging=False)
        allProcesses.append(processCamera)

    if BoschNet:
        processBoschNet = processBoschNet(queueList, logging, Speed, Steer, debugging=True)
        allProcesses.append(processBoschNet)
    
    # Initializing serial connection NUCLEO - > PI
    if SerialHandler:
        processSerialHandler = processSerialHandler(queueList, logging, Speed, Steer)
        allProcesses.append(processSerialHandler)

    if Server:
        hostname = args.xavierip # Xavier IP
        port = args.xavierport
        kindofimages = ["Position", "ObjectDetection", "Points", "Segmentation"]
        kind = kindofimages[1]
        processServer1 = processServer(
            queueList, logging, hostname, port, kind, debugging=False
        )
        allProcesses.append(processServer1)
        port += 1
        kind = kindofimages[2]
        processServer2 = processServer(
            queueList, logging, hostname, port, kind, debugging=False
        )
        allProcesses.append(processServer2)
        port += 1
        kind = kindofimages[0]
        processServer3 = processServer(
            queueList, logging, hostname, port, kind, debugging=False
        )
        allProcesses.append(processServer3)

    if ClearBuffer:
        processClearBuffer = processClearBuffer(queueList, logging, debugging=False)
        allProcesses.append(processClearBuffer)

    # if DecisionMaking:
    #     processDecisionMaking = processDecisionMaking(queueList, logging, Speed, Steer, debugging=True)
    #     allProcesses.append(processDecisionMaking)

    if ManualControl:
        processManualControl = processManualControl(
            queueList, logging, Speed, Steer, debugging=False
        )
        allProcesses.append(processManualControl)
    
    if TrafficCommunication:
        processTrafficCommunication = processTrafficCommunication(queueList, logging, 3, debugging=False)
        allProcesses.append(processTrafficCommunication)

    # ===================================== START PROCESSES ==================================
    for process in allProcesses:
        process.daemon = True
        process.start()

    # ===================================== STAYING ALIVE ====================================
    blocker = Event()
    try:
        blocker.wait()
    except KeyboardInterrupt:
        print("\nCatching a Keyboard Interruption exception! Shutdown all processes.\n")
        for proc in allProcesses:
            print("Process stopped", proc)
            proc.stop()
            time.sleep(0.1)
            proc.join()
