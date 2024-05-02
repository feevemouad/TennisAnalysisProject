from utils import (read_video, save_video, compute_homography)
from trackers import PlayerBallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2


def main():
    # Read Video
    input_video_path = r"video\30s.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_ball_tracker = PlayerBallTracker(model_path=r"models\player_and_ball_best.pt")

    player_ball_detections = player_ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_ball_detections.pkl"
                                                     )
    

    player_ball_detections = player_ball_tracker.interpolate_ball_positions(player_ball_detections)
    
    
    # # Court Line Detector model
    court_line_detector = CourtLineDetector(r"models\keypoints_1_69.model.h5")
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 


    # Draw output
    
    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    ## Draw Player and Ball Bounding Boxes
    output_video_frames= player_ball_tracker.draw_bboxes(output_video_frames, player_ball_detections)
    ## Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    ## Draw Players and Ball on the Mini Court
    
    court_four_keypoints = list(zip(court_keypoints[0:8:2],court_keypoints[1:8:2]))
    mini_court_four_keypoints = list(zip(mini_court.drawing_key_points[0:8:2], mini_court.drawing_key_points[1:8:2]))
    
    homography_matrix = compute_homography( court_four_keypoints, mini_court_four_keypoints)
    
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_ball_detections, homography_matrix)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    save_video(output_video_frames, "video/output_video.avi")

if __name__ == "__main__":
    main()  