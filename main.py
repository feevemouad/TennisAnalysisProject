from utils import (read_video, save_video, compute_homography)
from trackers import PlayerBallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import streamlit as st
import os
from tempfile import NamedTemporaryFile
from moviepy.editor import VideoFileClip

def release_and_delete_file(path):
    """Attempt to release and delete a file, retrying if it fails."""
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            os.unlink(path)
            break
        except PermissionError:
            time.sleep(1)  # Wait for 1 second before retrying
        if attempt == max_attempts - 1:
            st.error("Could not delete file after multiple attempts.")

def process_video(input_video_path, output_video_path):
    video_frames = read_video(input_video_path)

    player_ball_tracker = PlayerBallTracker(model_path="models/player_and_ball_best.pt")
    player_ball_detections = player_ball_tracker.detect_frames( video_frames,
                                                                read_from_stub=False,
                                                                stub_path="tracker_stubs/player_ball_detections.pkl"
                                                                )
    player_ball_detections = player_ball_tracker.interpolate_ball_positions(player_ball_detections)

    court_line_detector = CourtLineDetector("models/keypoints_1_69.model.h5")
    court_keypoints = court_line_detector.predict(video_frames[0])
    mini_court = MiniCourt(video_frames[0])

    output_video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    output_video_frames = player_ball_tracker.draw_bboxes(output_video_frames, player_ball_detections)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    court_four_keypoints = list(zip(court_keypoints[0:8:2], court_keypoints[1:8:2]))
    mini_court_four_keypoints = list(zip(mini_court.drawing_key_points[0:8:2], mini_court.drawing_key_points[1:8:2]))
    homography_matrix = compute_homography(court_four_keypoints, mini_court_four_keypoints)

    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_ball_detections, homography_matrix)
    
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    save_video(output_video_frames, output_video_path)

    # Reencode the video using moviepy with h264 codec
    clip = VideoFileClip(output_video_path)
    clip.write_videofile(output_video_path[:-4] + "_h264.mp4", codec='libx264')

def main():
    # Streamlit UI
    st.title("Tennis Analysis App")

    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

    if video_file is not None:
        st.video(video_file)
        
        if st.button("Process Video"):
            with st.spinner('Processing video...'):
                with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input_file, NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_file:
                    tmp_input_file.write(video_file.read())
                    tmp_input_file.flush()  # Ensure all data is written
                    os.fsync(tmp_input_file.fileno())  # Ensure all data is written to disk

                    process_video(tmp_input_file.name, tmp_output_file.name)

                    # Display processed video
                    st.subheader("Processed Video")
                    st.video(tmp_output_file.name[:-4] + "_h264.mp4")

                release_and_delete_file(tmp_input_file.name)
                release_and_delete_file(tmp_output_file.name)
                
                
if __name__ == "__main__":
    main()