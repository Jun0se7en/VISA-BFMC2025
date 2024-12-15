# ===================================== GENERAL IMPORTS ==================================
import sys

sys.path.append(".")
from multiprocessing import Queue, Event
import logging


# ===================================== PROCESS IMPORTS ==================================
from src.gateway.processGateway import processGateway
from src.hardware.camera.processCamera import processCamera
from src.clearBuffer.processClearBuffer import processClearBuffer
from src.server.processServer import processServer
from src.laneDetection.processSegmentation import processSegmentation

# ======================================== SETTING UP ====================================
allProcesses = list()
queueList = {
    "Critical": Queue(),
    "Warning": Queue(),
    "General": Queue(),
    "Config": Queue(),    
    "Camera": Queue(),
    "Points": Queue(),
}

logging = logging.getLogger()

Segmentation = False

Camera = False

Server = False

ClearBuffer = False
# ===================================== SETUP PROCESSES ==================================

# Initializing Gateway
processGateway = processGateway(queueList, logging)
allProcesses.append(processGateway)

# Initializing Camera
if Camera:
    processCamera = processCamera(queueList, logging)
    allProcesses.append(processCamera)

# Initializing Server
if Server:
    hostname = "192.168.20.1"
    port = 1234
    processServer = processServer(
        queueList, logging, hostname, port, debugging=False
    )
    allProcesses.append(processServer)

if ClearBuffer:
    processClearBuffer = processClearBuffer(queueList, logging, debugging=False)
    allProcesses.append(processClearBuffer)

if Segmentation:
    processSegmentation = processSegmentation(queueList, logging, debugging=False)
    allProcesses.append(processSegmentation)
    

# ===================================== START PROCESSES ==================================
for process in allProcesses:
    process.daemon = True
    process.start()

# ===================================== STAYING ALIVE ====================================
blocker = Event()
try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        print("Process stopped", proc)
        proc.stop()
        proc.join()
