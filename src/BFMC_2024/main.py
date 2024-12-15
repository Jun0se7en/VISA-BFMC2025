import sys

sys.path.append(".")
import logging
import os
import time
from multiprocessing import Event, Queue
from multiprocessing.sharedctypes import Value

import torch

from src.clearBuffer.processClearBuffer import processClearBuffer
from src.control.imageControl.processImageControl import processImageControl
from src.control.manualControl.processManualControl import processManualControl

# from src.data.CarsAndSemaphores.processCarsAndSemaphores import processCarsAndSemaphores
from src.data.TrafficCommunication.processTrafficCommunication import (
    processTrafficCommunication,
)
from src.directionControl.decisionMaking.processDecisionMaking import (
    processDecisionMaking,
)

# ===================================== PROCESS IMPORTS ==================================
from src.gateway.processGateway import processGateway
from src.hardware.camera.processCamera import processCamera
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.imageProcessing.laneDetection.processSegmentation import processSegmentation
from src.imageProcessing.objectDetection.processObjectDetection import (
    processObjectDetection,
)
from src.position_fusion.processPositionFusion import processPositionFusion
from src.server.processServer import processServer
from src.utils.PCcommunicationDashBoard.processPCcommunication import (
    processPCCommunicationDashBoard,
)
from src.utils.PCcommunicationDemo.processPCcommunication import (
    processPCCommunicationDemo,
)

from src.piCamera.processpiCamera import processpiCamera

# ======================================== SETTING UP ====================================
allProcesses = list()
queueList = {
    'Control': Queue(),
    "Critical": Queue(),
    "Warning": Queue(),
    "General": Queue(),
    "Config": Queue(),
    # After Processed
    "Segmentation": Queue(),
    "ObjectDetection": Queue(),
    "MiddlePoint": Queue(),
    "ObjectDetectionImage": Queue(),
    
    # Message for Lane Keeping
    "LaneDetection": Queue(),
    "Intersection": Queue(),
    "Points": Queue(),
    # Camera
    "MainCamera": Queue(),
    "ObjectCamera": Queue(),
    "SegmentCamera": Queue(),
    # Speed
    "Steer": Queue(),
    "Speed": Queue(),
    "Test": Queue(),
}

Speed = Value("i", 0)
Steer = Value("i", 0)

logging = logging.getLogger()

# Pi Camera
PiCamera = True

# Camera
Camera = False

# Just false all when running
TrafficCommunication = False
PCCommunicationDemo = False
PCCommunicationDashboard = False
CarsAndSemaphores = False

# Serial Handler
SerialHandler = True
  
# Image Processing
Segmentation = False
ObjectDetection = True

# Server
Server = True

# WASD control
ManualControl = False

# CV control
DecisionMaking = True

# Position Fusion
PositionFusion = False

# Clear Buffer (Optional Remove Further)
ClearBuffer = False

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
    width = 320.0
    height = 240.0
    processCamera = processCamera(queueList, logging, width, height, debugging=False)
    allProcesses.append(processCamera)

# Initializing interface
if PCCommunicationDemo:
    processPCCommunication = processPCCommunicationDemo(queueList, logging)
    allProcesses.append(processPCCommunication)
if PCCommunicationDashboard:
    processPCCommunicationDashBoard = processPCCommunicationDashBoard(
        queueList, logging
    )
    allProcesses.append(processPCCommunicationDashBoard)

# # Initializing cars&sems
# if CarsAndSemaphores:
#     processCarsAndSemaphores = processCarsAndSemaphores(queueList)
#     allProcesses.append(processCarsAndSemaphores)

# Initializing GPS
if TrafficCommunication:
    processTrafficCommunication = processTrafficCommunication(queueList, logging, 3)
    allProcesses.append(processTrafficCommunication)

# Initializing serial connection NUCLEO - > PI
if SerialHandler:
    processSerialHandler = processSerialHandler(queueList, logging, Speed, Steer)
    allProcesses.append(processSerialHandler)

if ObjectDetection:
    library = "./models/v8/libmyplugins.so"
    engine = "./models/v8/08-03-2024v8.engine"
    conf_thres = 0.6
    iou_thres = 0.4
    classes = [
        "Car",
        "CrossWalk",
        "Greenlight",
        "HighwayEnd",
        "HighwayEntry",
        "NoEntry",
        "OneWay",
        "Parking",
        "Pedestrian",
        "PriorityRoad",
        "Redlight",
        "Roundabout",
        "Stop",
        "Yellowlight",
    ]
    # classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    #         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    #         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    #         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    #         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    #         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    #         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    #         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    #         "hair drier", "toothbrush"]
    processObjectDetection = processObjectDetection(
        queueList,
        logging,
        library,
        engine,
        conf_thres,
        iou_thres,
        classes,
        debugging=True,
    )
    allProcesses.append(processObjectDetection)

if Segmentation:
    processSegmentation = processSegmentation(queueList, logging, debugging=False)
    allProcesses.append(processSegmentation)

if Server:
    hostname = "192.168.7.233" # TX2 IP
    port = 12345
    kindofimages = ["MainCamera", "ObjectDetectionImage", "MiddlePoint"]
    kind = kindofimages[2]
    processServer1 = processServer(
        queueList, logging, hostname, port, kind, debugging=False
    )
    allProcesses.append(processServer1)
    port += 1
    kind = kindofimages[1]
    processServer2 = processServer(
        queueList, logging, hostname, port, kind, debugging=False
    )
    allProcesses.append(processServer2)

# Initializing Manual Control
if ManualControl:
    processManualControl = processManualControl(
        queueList, logging, Speed, Steer, debugging=False
    )
    allProcesses.append(processManualControl)

if DecisionMaking:
    processDecisionMaking = processDecisionMaking(
        queueList, logging, Speed, Steer, debugging=False
    )
    allProcesses.append(processDecisionMaking)

if PositionFusion:
    processPositionFusion = processPositionFusion(
        queueList, Speed, Steer, logging, debugging=False
    )
    allProcesses.append(processPositionFusion)


if ClearBuffer:
    processClearBuffer = processClearBuffer(queueList, logging, debugging=False)
    allProcesses.append(processClearBuffer)

if PiCamera:
    SERVER_IP = '192.168.20.1' # Pi IP
    PORT = 1234
    processpiCamera = processpiCamera(serverip=SERVER_IP, port=PORT, queuesList=queueList, debugging=False)
    allProcesses.append(processpiCamera)

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
        proc.join()
