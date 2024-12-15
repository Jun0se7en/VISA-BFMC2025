import cv2
import threading
import base64
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    serialCamera,
    Points,
    Record,
    Config,
)
from src.templates.threadwithstop import ThreadWithStop
from src.laneDetection.utils import utils_action
from src.laneDetection import ImagePreprocessing
from src.laneDetection import IntersectionDetection
from src.laneDetection import LaneDetection

# Use this thread for LaneLine Segmentation
class threadSegmentation(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, debugger):
        super(threadSegmentation, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        pipeRecvCamera, pipeSendCamera = Pipe(duplex=False)
        self.pipeRecvCamera = pipeRecvCamera
        self.pipeSendCamera = pipeSendCamera
        self.subscribe()
        self.Configs()
        self._init_segment()
        # print('Initialize camera thread!!!')

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadSegmentation", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadSegmentation", "pipe": self.pipeSendConfig},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": serialCamera.Owner.value,
                "msgID": serialCamera.msgID.value,
                "To": {"receiver": "threadSegmentation", "pipe": self.pipeSendCamera},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        # cv2.destroyAllWindows()
        super(threadSegmentation, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    def display_points(self, points, image, color):     # For lane highlighting
        if color == 0:
            if points is not None:
                for point in points:
                    point_tp = tuple(point)
                    image = cv2.circle(image, point_tp, 1, (255, 0, 0), -1)    #blue
            return image
        if color == 1:
            if points is not None:
                for point in points:
                    point_tp = tuple(point)
                    image = cv2.circle(image, point_tp, 1, (0, 0, 255), -1)    #red
            return image
    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            start = time.time()
            # if self.pipeRecvCamera.poll():
            #     msg = self.pipeRecvCamera.recv()
                # msg = msg['value']
            if not self.queuesList['Camera'].empty():
                msg = self.queuesList['Camera'].get()
                msg = msg["msgValue"]
                image_data = base64.b64decode(msg)
                img = np.frombuffer(image_data, dtype=np.uint8)     
                image = cv2.imdecode(img, cv2.IMREAD_COLOR)
                check_thresh = self.opt['INTERSECT_DETECTION']
                crop_ratio = float(check_thresh['crop_ratio'])
                height = self.opt["IMAGE_SHAPE"]["height"]

                crop_height_value =  int(height * crop_ratio)
                im_cut = image[crop_height_value:, :]   # crop half of image for intersection det
                # Intersection detection
                hlane_det = self.ImageProcessor.process_image2(im_cut)
                check_intersection = self.IntersectFinder.detect(hlane_det)
                

                # Lane detection
                new_im = np.copy(image)
                lane_det, grayIm = self.ImageProcessor.process_image(image)
                left_points, right_points, _, _ = self.LaneLine.find_left_right_lane(lane_det)  # Find left, right laneline

                self.queuesList[Points.Queue.value].put(
                {
                    "Owner": Points.Owner.value,
                    "msgID": Points.msgID.value,
                    "msgType": Points.msgType.value,
                    "msgValue": {'Left': left_points, 'Right': right_points, 'Intersection': check_intersection[0], 'Image': msg, 'Sending Time': start},
                })
                

    # =============================== START ===============================================
    def start(self):
        super(threadSegmentation, self).start()

    def _init_segment(self):
        self.opt = utils_action.load_config_file("main_rc.json")
        self.ImageProcessor = ImagePreprocessing.ImagePreprocessing(self.opt)
        self.IntersectFinder = IntersectionDetection.IntersectionDetection(self.opt, debug=True)
        self.LaneLine = LaneDetection.LaneDetection(self.opt)