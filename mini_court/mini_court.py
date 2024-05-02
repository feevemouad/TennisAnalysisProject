import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 180
        self.drawing_rectangle_height = 360 
        self.buffer = 30
        self.padding_court=15

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.buffer
    
    def draw_court(self, frame):
        line_color = (235, 235, 235) 
        net_color = (255, 102, 50)  
        point_color = (255, 255, 255)  

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, line_color, 2)  # Increased line thickness

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, net_color, 2, lineType=cv2.LINE_AA)
        # Optional: draw a dashed line for the net
        for i in range(int(net_start_point[0]), int(net_end_point[0]), 10):
            if (i // 10) % 2 == 0:
                cv2.line(frame, (i, net_start_point[1]), (i+5, net_start_point[1]), net_color, 2, lineType=cv2.LINE_AA)

        # draw points
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x, y), 3, point_color, -1)  # Slightly larger circles

        return frame

    def draw_background_rectangle(self, frame):
        # Initialize a blank mask the size of the frame
        shapes = np.zeros_like(frame, np.uint8)

        # Constants for the rectangle and corners
        corner_radius = 20  # Radius of the rounded corners
        color = (0,102,50)  # Dark gray fill
        thickness = -1  # Fill the rectangle

        # Calculate coordinates for simplicity
        top_left = (self.start_x, self.start_y)
        bottom_right = (self.end_x, self.end_y)

        # Draw the rectangle with rounded corners
        cv2.rectangle(shapes, (top_left[0] + corner_radius, top_left[1]),
                    (bottom_right[0] - corner_radius, bottom_right[1]), color, thickness)
        cv2.rectangle(shapes, (top_left[0], top_left[1] + corner_radius),
                    (bottom_right[0], bottom_right[1] - corner_radius), color, thickness)

        # Drawing four ellipse corners to create the rounded effect
        cv2.ellipse(shapes, (top_left[0] + corner_radius, top_left[1] + corner_radius),
                    (corner_radius, corner_radius), 180, 0, 90, color, thickness)
        cv2.ellipse(shapes, (bottom_right[0] - corner_radius, top_left[1] + corner_radius),
                    (corner_radius, corner_radius), 270, 0, 90, color, thickness)
        cv2.ellipse(shapes, (top_left[0] + corner_radius, bottom_right[1] - corner_radius),
                    (corner_radius, corner_radius), 90, 0, 90, color, thickness)
        cv2.ellipse(shapes, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
                    (corner_radius, corner_radius), 0, 0, 90, color, thickness)

        # Blend the shape onto the original frame
        out = frame.copy()
        alpha = 0.3  # Transparency factor
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
        
    def apply_homography(self, homography_matrix, points):
        points = np.array(points, dtype="float32")
        transformed_points = cv2.perspectiveTransform(np.array([points]), homography_matrix)
        return transformed_points[0]

    def draw_points_on_mini_court(self, frames, player_ball_detections, homography_matrix):
        # Iterate over frames and detections
        for frame, detections in zip(frames, player_ball_detections):
            points = []
            ids = []
            for id, bbox in detections.items():
                if id in [0, 1]:  # Player front and back
                    # Use the bottom center of the bbox for players
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = bbox[3] - 6  # bottom y-coordinate
                elif id == 2:  # Tennis ball
                    # Use centroid for the ball
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                
                points.append((x_center, y_center))
                ids.append(id)
            
            if points:
                # Transform points using the homography matrix
                transformed_points = self.apply_homography(homography_matrix, points)
                
                # Draw each point on the frame
                for point, id in zip(transformed_points, ids):
                    x, y = int(point[0]), int(point[1])
                    if id in {0,1}:
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  
                    else : 
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  
                    
        return frames
                
            
            
        
