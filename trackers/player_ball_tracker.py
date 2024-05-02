from ultralytics import YOLO 
import cv2
import pickle
import sys
import pandas as pd

sys.path.append('../')

class PlayerBallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    def interpolate_ball_positions(self, player_ball_detections):
        ball_positions = [x.get(2,[]) for x in player_ball_detections]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = df_ball_positions.to_numpy().tolist()

        for i in range(len(player_ball_detections)):
            player_ball_detections[i][2] = ball_positions[i]

        return player_ball_detections

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_ball_detections = pickle.load(f)
            return player_ball_detections

        for frame in frames:
            player_ball_dict = self.detect_frame(frame)
            player_ball_detections.append(player_ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_ball_detections, f)
        
        return player_ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame, conf=0.2)[0]
        player_ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            player_ball_dict[object_cls_id] = result
        
        return player_ball_dict

    def draw_bboxes(self,video_frames, player_ball_detections):
        output_video_frames = []
        for frame, player_ball_dict in zip(video_frames, player_ball_detections):
            # Draw Bounding Boxes
            for id, bbox in player_ball_dict.items():
                x1, y1, x2, y2 = bbox
                if id in {0,1} :
                    cv2.putText(frame, "Front Player" if id==0 else "Back Player",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 0, 0), 2)
                if id == 2:
                    cv2.putText(frame, f"Ball",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
