import numpy as np
import math
import cv2
from collections import defaultdict
from src.laneDetection.utils import utils_action as action
from src.laneDetection import ImagePreprocessing
class LaneDetection:
    def __init__(self, opt):
        self.opt = opt["LANE_KEEPING"]
        self.im_height = opt["IMAGE_SHAPE"]["height"]
        self.im_width = opt["IMAGE_SHAPE"]["width"]
    
    def get_middle_point(self, points):
        if len(points) > 0:
            points = np.array(points)
            mean_x = np.mean(points[:, 0])
            anchor_idx = np.argmin(np.abs(mean_x - points[:, 0]))                           # find close point to the mean

            middle_point = points[anchor_idx, :]                                
            middle_points = points[np.abs(middle_point[0] - points[:, 0]) < 30]             # For point have the distance to the mean point > 30, discard it
            middle_point = np.mean(middle_points, axis = 0, dtype=np.int32)                 # The middle point is calculated by the mean of the remaining point
            return middle_point, middle_points
        
        else:
            return None, None
        
    def find_left_right_lane(self, bin_image):
        pixel_coordinates = np.argwhere(bin_image == 255)[:, ::-1]  # Find white pixels coordinates
        pixel_coordinates = pixel_coordinates[np.argsort(pixel_coordinates[::, 0])][::-1]   # Sort them in order of x coordinate
        y_mapping = defaultdict(list)
        for point in pixel_coordinates:
            y_mapping[point[1]].append(point) # Key = y coordinate
            
        lowest_left_pixel, lowest_right_pixel = [], []
        # Find anchor of each lane, if anchor is missing -> raise state
        for anchor_height in range(self.im_height - self.opt["limit_sweep"] -1, 0, self.opt["anchor_step"]):
            if y_mapping[anchor_height] != []:
                points = y_mapping[anchor_height]
                for point in points:
                    if point[0] <  self.im_width //2:                      
                        lowest_left_pixel.append(point)
                    elif point[0] > self.im_width //2:
                        lowest_right_pixel.append(point)
                # Find if the highest pixel is in right or left lane        
                if len(lowest_left_pixel) > 0 or len(lowest_right_pixel) > 0:
                    break
                
        left_anchor, _ = self.get_middle_point(lowest_left_pixel)
        right_anchor, _ = self.get_middle_point(lowest_right_pixel)
        if left_anchor is None:                                                             # If we could not find left or right point, init the dummy point
            left_anchor = [0, anchor_height]               

        if right_anchor is None:
            right_anchor = [self.im_width, anchor_height]

        left_points = []
        right_points = []

        # From anchor height, we traverse back to the top of the image
        # we separate left and right point based on the distance to left anchor and right anchor.

        for height in range(anchor_height - 1, 120, self.opt["step"]):
            if y_mapping[height] != []:
                if right_anchor[0] == self.im_width:
                    right_anchor[1] = height

                if left_anchor[0] == -1:
                    left_anchor[1] = height

                points = y_mapping[height]
                current_left_anchor = []
                current_right_anchor = []

                for point in points:
                    # calculate the distance to the left anchor and to the right anchor
                    left_offset = abs(point[0] - left_anchor[0]) * self.opt["x_ratio"] \
                                    + abs(point[1] - left_anchor[1]) * self.opt["y_ratio"]    
                    
                    right_offset = abs(point[0] - right_anchor[0]) * self.opt["x_ratio"] \
                                    + abs(point[1] - right_anchor[1]) * self.opt["y_ratio"]

                    if left_offset < right_offset and abs(point[1] - left_anchor[1]) < self.opt["y_dist"]:
                        if  abs(point[0] - left_anchor[0]) < self.opt["x_dist"]:                # if anchor is not dummy anchor, we compare x_axis
                            current_left_anchor.append(point)
                        elif left_anchor[0] == -1:
                            current_left_anchor.append(point)

                    elif left_offset > right_offset and abs(point[1] - right_anchor[1]) < self.opt["y_dist"]:
                        if  abs(point[0] - right_anchor[0]) < self.opt["x_dist"]:             # if anchor is not dummy anchor, we compare x_axis
                            current_right_anchor.append(point)
                        elif right_anchor[0] == self.im_width:
                            current_right_anchor.append(point)
                        # else:
                        #     current_right_anchor.append(point)
                # find the middle point
                left_middle_point, left_middle_points = self.get_middle_point(current_left_anchor)
                right_middle_point, right_middle_points = self.get_middle_point(current_right_anchor)

                # update the new anchor point for left and right lane 
                if left_middle_point is not None:
                    left_anchor = left_middle_point
                    for point in left_middle_points:
                        left_points.append(point)
                
                if right_middle_point is not None:
                    right_anchor = right_middle_point
                    for point in right_middle_points:
                        right_points.append(point)
        
        return np.array(left_points), np.array(right_points), left_anchor, right_anchor
    
if __name__ == '__main__':
    def display_points(points, image):
        if points is not None:
            image = cv2.circle(image, points, 5, (255, 0, 0), -1)
        return image
    im = cv2.imread(r'test_real/image50.jpg')
    opt = action.load_config_file("main_rc.json")
    direct = LaneDetection(opt)
    im_pros = ImagePreprocessing.ImagePreprocessing(opt)
    lane = im_pros.process_image(im)
    a, b, c, d = direct.find_left_right_lane(lane)
    for i in a:
        dis_im = cv2.circle(im, i, 1, (255, 0, 0), -1)
    for i in b:
        dis_im = cv2.circle(im, i, 1, (0, 0, 255), -1)
    dis_im = cv2.circle(im, tuple(c), 2, (0, 255, 0), -1)
    dis_im = cv2.circle(im, tuple(d), 2, (0, 255, 0), -1)
    cv2.imshow('a', dis_im)
    cv2.imshow('b', lane)
    cv2.waitKey(0)
        