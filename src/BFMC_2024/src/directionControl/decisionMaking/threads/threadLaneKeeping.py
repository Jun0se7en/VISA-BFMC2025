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
    Record,
    Config,
    SpeedMotor,
    SteerMotor,
    MiddlePoint,
)
from src.templates.threadwithstop import ThreadWithStop
from src.directionControl.decisionMaking.threads.SignTrafficHandler import SignTrafficHandler
from src.utils.CarControl.CarControl import CarControl
class threadLaneKeeping(ThreadWithStop):
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
        super(threadLaneKeeping, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.Speed = Speed
        self.Steer = Steer
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        self.speed = 0
        self.angle = 0
        self.debugger = debugger
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord

        self.subscribe()
        self.Configs()
        self.message = {}
        self.message_type = ""
        
        self.TrafficHandler = SignTrafficHandler(self.queuesList, self.Speed, self.Steer)
    # 2 states of motors
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
        self.speed = 0
        self.angle = 0
        self.Queue_Sending()
        super(threadLaneKeeping, self).stop()

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
        self.SPEED = 40
        self.prev_obj = ''
        self.extra_speed = 0
        self.swap_event = False
        self.obj_area_check = False
        self.enable_inter = False
        self.enable_pedestrian = False
        self.enable_light = False
        # self.inter_msg = 0
        self.prev_inter = 0
        self.area = 0
        self.obj_msg = []
        self.obj_assigned = ''
        self.flag_highway = False
        self.flag_end = False
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            start = time.time()
            if self.swap_event == False:
                # Check obj state
                if not self.queuesList["ObjectDetection"].empty():
                    self.obj_msg = self.queuesList["ObjectDetection"].get() 
                    self.area = self.obj_msg['msgValue']['Area']
                    self.obj_msg = self.obj_msg["msgValue"]["Class"]

                
                    if len(self.obj_msg) == 1:
                        # print('present: ',self.obj_msg)
                        # print('past: ', self.prev_obj)
                        self.obj_msg = self.obj_msg[0]
                        self.area = self.area[0]
                        print('Object: ', self.obj_msg)
                        print('Area: ', self.area)
                        if self.TrafficHandler.check_special_sign(self.obj_msg, self.area) == True:
                            if self.obj_msg == 'HighwayEntry':
                                self.extra_speed = 7
                            elif self.obj_msg == 'HighwayEnd':
                                self.extra_speed = -2
                            elif self.obj_msg == 'CrossWalk':
                                self.flag_end = True
                                self.extra_speed = -10
                            elif self.obj_msg == 'Pedestrian':
                                self.extra_speed = -self.SPEED
                                self.enable_pedestrian = True
                                # print('pedestrian ', self.enable_pedestrian)
                            elif self.obj_msg == 'Yellowlight':
                                # print('yellow light!!!')
                                self.extra_speed = -10
                            elif self.obj_msg == 'Redlight':
                                self.extra_speed = -self.SPEED
                                # print('redlight ', self.enable_light)
                                self.enable_light = True
                            elif self.obj_msg == 'Greenlight' and self.enable_light == True:
                                # print('greenLight: ', self.enable_light)
                                self.extra_speed = -5
                                self.enable_light = False
                            elif self.obj_msg == 'Parking':
                                self.TrafficHandler.swap_case(self.obj_msg)
                            elif self.obj_msg == 'Roundabout':
                                self.TrafficHandler.swap_case(self.obj_msg)
                            elif self.obj_msg == 'Car':
                                self.TrafficHandler.swap_case(self.obj_msg)
                            elif self.obj_msg == 'NoEntry':
                                print('no entry')
                                # self.flag_end = True

                                self.TrafficHandler.swap_case(self.obj_msg)
                            if self.prev_obj != self.obj_msg:
                                self.prev_obj = self.obj_msg
                            print(self.obj_msg, ' ', self.area)
                                
                        else:   
                            if self.TrafficHandler.check_area(self.obj_msg, self.area) == True:
                                self.obj_assigned = self.obj_msg
                                if self.prev_obj != self.obj_msg:
                                    self.prev_obj = self.obj_msg
                                self.extra_speed = -5
                                self.obj_area_check = True
                            

                    # elif len(self.obj_msg) == 2:
                    #     if 'Pedestrian' in self.obj_msg:
                    #         self.extra_speed = -28
                    else: 
                        if self.enable_pedestrian == True and len(self.obj_msg) == 0:
                            self.extra_speed = 0
                            self.enable_pedestrian = False
                            

                         
                    if self.obj_area_check == True and len(self.obj_msg) == 0:
                        self.swap_event = True
                        self.extra_speed = 0

                        
                # start = time.time()      
                if not self.queuesList["Points"].empty():

                    message = self.queuesList["Points"].get()
                    message = message['msgValue']
                    left_points = list()
                    right_points = list()
                    left_points = message['Left']
                    right_points = message['Right']
                    image_data = base64.b64decode(message['Image'])
                    img = np.frombuffer(image_data, dtype=np.uint8)
                    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    # print(len(right_points))
                    # print(len(left_points))

                    ## Finding Middle Lane
                    black_image = np.copy(image)
                    middle_points = list()
                    num_points = 80 # number of points
                    angle = 0
                    if(left_points.shape[0] < num_points and right_points.shape[0] < num_points):
                        angle = 0
                    elif(left_points.shape[0] >= num_points and right_points.shape[0] >= num_points):
                        # print('Two Lines')
                        # Finding left function
                        x_lefts = list()
                        y_lefts = list()
                        for i in left_points:
                            x_lefts.append(i[0])
                            y_lefts.append(i[1])
                        left_func = np.polyfit(x_lefts,y_lefts,2)
                        # print('Left Func:',left_func)
                        # Finding right function
                        x_rights = list()
                        y_rights = list()
                        for i in right_points:
                            x_rights.append(i[0])
                            y_rights.append(i[1])
                        right_func = np.polyfit(x_rights,y_rights,2)
                        # print('Right Func:',right_func)

                        # middle_points = list()
                        for i in range(239, 99, -1):
                            ######################################################################
                            x_left = np.roots([left_func[0], left_func[1], left_func[2]-i])
                            x1 = 99999
                            for j in x_left:
                                if j>0 and x1>j and j > 0:
                                    x1 = j

                            # x1 = min(x_left[0], x_left[1])
                            # if (np.abs(np.imag(x1)) < 1e-10) > 0:
                                cv2.circle(black_image, (np.int64(x1), i), 5, (255,0,0), -1)
                                # print(f'x: {x1}, y: {i}')
                            
                            ######################################################################
                            x_right = np.roots([right_func[0], right_func[1], right_func[2]-i])
                            x2 = -1
                            for j in x_right:
                                if j>0 and x2<j and j<320:
                                    x2 = j
                            # x2 = min(x_right[0], x_right[1])
                            # if (np.abs(np.imag(x2)) < 1e-10) > 0:
                                cv2.circle(black_image, (np.int64(x2), i), 5, (0,0,255), -1)
                                # print(f'x: {x2}, y: {i}')

                            if ((np.abs(np.imag(x1)) < 1e-10) > 0 and (np.abs(np.imag(x2)) < 1e-10) > 0):
                                middle_point = (x1+x2)/2
                                cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,0), -1)
                                middle_points.append([np.int64(middle_point),i])
                                # print('STT: ', count+1, 'Left: ', x1, 'Middle: ', middle_point, 'Right: ', x2)
                        # print('Left', x_lefts)
                        # print('Right', x_rights)
                        # print('Middle', middle_points)
                    ## Finding Only Right Lane
                    elif left_points.shape[0]<num_points:
                        # print('Only Right Lane')
                        Right_Offset = -145
                        x_rights = list()
                        y_rights = list()
                        for i in right_points:
                            x_rights.append(i[0])
                            y_rights.append(i[1])
                        right_func = np.polyfit(x_rights,y_rights,2)
                        # print('Right Func: ', right_func)

                        # middle_points = list()
                        for i in range(239, 99, -1):
                            x_right = np.roots([right_func[0], right_func[1], right_func[2]-i])
                            x2 = -1
                            for j in x_right:
                                if j>0 and x2<j and j<320:
                                    x2 = j
                            # x2 = max(x_right[0], x_right[1])
                            if (np.abs(np.imag(x2)) < 1e-10) > 0:
                                cv2.circle(black_image, (np.int64(x2), i), 5, (0,0,255), -1)
                                # print(f'x: {x2}, y: {i}')
                                #################################################
                                middle_point = np.int64(x2+Right_Offset-0.55*(i-240+1))
                                cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,0), -1)
                                middle_points.append([np.int64(middle_point),i])
                    ## Finding Only Left Lane
                    elif right_points.shape[0]<num_points:
                        # print('Only Left Lane')
                        Left_Offset = 145
                        x_lefts = list()
                        y_lefts = list()
                        for i in left_points:
                            x_lefts.append(i[0])
                            y_lefts.append(i[1])
                        left_func = np.polyfit(x_lefts,y_lefts,2)
                        # print('Left Func: ', left_func)


                        # middle_points = list()
                        for i in range(239, 99, -1):
                            x_left = np.roots([left_func[0], left_func[1], left_func[2]-i])
                            x1 = 99999
                            for j in x_left:
                                if j>0 and x1>j and j<320:
                                    x1 = j
                            # x1 = min(x_left[0], x_left[1])
                            if (np.abs(np.imag(x1)) < 1e-10) > 0:
                                cv2.circle(black_image, (np.int64(x1), i), 5, (255,0,0), -1)
                                # print(f'x: {x2}, y: {i}')
                                ########################################
                                middle_point = np.int64(x1+Left_Offset+0.55*(i-240+1))
                                cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,0), -1)
                                middle_points.append([np.int64(middle_point),i])
                    # print(middle_points)
                    # Calculate Angle
                    chosen_middle = 150
                    # angle_thres = 3
                    
                    if len(middle_points)>0:
                        ## Only < chosen middle
                        if middle_points[0][1] < chosen_middle:
                            x = middle_points[0][0]
                            y = middle_points[0][1]
                            angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                            angle = np.int64(angle/2.3)
                            # print(x, y)
                            cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                        ## Only > chosen middle
                        elif middle_points[-1][1] > chosen_middle:
                            x = middle_points[-1][0]
                            y = middle_points[-1][1]
                            angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                            if(y > chosen_middle and y < chosen_middle + 23): # thres 1
                                # print('Thres 1')
                                angle = np.int64(angle/2)
                            elif(y >= chosen_middle + 23 and y < chosen_middle + 23*2): # thres 2
                                # print('Thres 2')
                                angle = np.int64(angle/2)
                            elif(y >= chosen_middle + 23*2 and y < chosen_middle + 23*2 + 22): # thres 3
                                # print('Thres 3')
                                angle = np.int64(angle/2)
                            elif(y >= chosen_middle + 23*2+22 and y < chosen_middle + 23*2 + 22*2): # thres 4
                                # print('Thres 4')
                                angle = np.int64(angle/2)
                            # print(x, y)
                            cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                        else:
                            for i in sorted(middle_points, key=lambda x: x[1]):
                                if i[1] == chosen_middle:
                                    x = i[0]
                                    y = i[1]
                                    angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi)) 
                                    angle = np.int64(angle/2.3)
                                    cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                                    # print(x, y)
                                    break
                            
                    # Send Image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 3
                    text_color = (0, 0, 255)  # Màu văn bản: Trắng
                    text_position = (10, 20)  # Vị trí của văn bản: Góc trái phía trên
                    cv2.putText(black_image, str(angle), text_position, font, font_scale, text_color, font_thickness)
                    # print('Lane Keeping: ', time.time() - start)
                    _, encoded_img = cv2.imencode(".jpg", black_image)
                    image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                    self.queuesList[MiddlePoint.Queue.value].put(
                    {
                        "Owner": MiddlePoint.Owner.value,
                        "msgID": MiddlePoint.msgID.value,
                        "msgType": MiddlePoint.msgType.value,
                        "msgValue": image_data_encoded,
                    })
                    
                    # Give speed, angle to motor
                    # print(self.extra_speed)
                    self.speed = self.SPEED + self.extra_speed
                    # self.speed = 25 
                    self.angle = -angle
                    # print('Speed: ', self.speed)
                    # print('Kiet' ,self.angle)
                    self.Queue_Sending()
            else:
                if not self.queuesList["Intersection"].empty():
                    self.inter_msg = self.queuesList["Intersection"].get()
                    self.inter_msg = self.inter_msg["msgValue"]
                    # print(self.prev_inter, ' ', self.inter_msg[0])
                    if self.prev_inter != self.inter_msg[0]:
                        if self.prev_inter == 0:
                            self.prev_inter = self.inter_msg[0]
                        if self.prev_inter - self.inter_msg[0] >= 100:  # Check threshold gap
                            self.enable_inter = True
                            # print('intersection')
                            # if self.obj_assigned == 'Stop' and self.prev_obj == 'NoEntry':
                            if self.flag_end == True and self.obj_assigned == 'Stop':
                                print('special case stop')
                                self.TrafficHandler.stop_special()
                                self.flag_end = False
                                
                            else:
                                self.TrafficHandler.swap_case(self.obj_assigned)
                            self.flag_end = False
                            self.prev_inter = 0
                            self.inter_msg[0] = 0
                            self.enable_inter = False
                            self.swap_event = False
                            self.obj_area_check = False
                        else:
                            self.prev_inter = self.inter_msg[0]
                if not self.queuesList["Points"].empty():
                    message = self.queuesList["Points"].get()
                    message = message['msgValue']
                    image = message['Image']
                    # _, encoded_img = cv2.imencode(".jpg", image)
                    # image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                    self.queuesList[MiddlePoint.Queue.value].put(
                    {
                        "Owner": MiddlePoint.Owner.value,
                        "msgID": MiddlePoint.msgID.value,
                        "msgType": MiddlePoint.msgType.value,
                        "msgValue": image_data_encoded,
                    })
            # print('Lane Keeping FPS: ', 1/(time.time() - start))
        



    # =============================== START ===============================================
    def start(self):
        super(threadLaneKeeping, self).start()

        
