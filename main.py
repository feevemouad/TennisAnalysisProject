import streamlit as st
import cv2
import os
import time
from utils import (read_video, save_video, compute_homography, icon)
from moviepy.editor import VideoFileClip
from tempfile import NamedTemporaryFile
from trackers import PlayerBallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

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

    court_line_detector = CourtLineDetector("models/keypoints_ 0_3093loos.model.h5")
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

def configure_sidebar():
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Yo fam! Start here ↓**", icon="👋🏾")
            st.markdown('Start analyzing tennis videos with ease! Simply upload your videos and explore a wealth of insights to improve your game.')
            video_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
            submitted = st.form_submit_button(
                    "Submit", type="primary", use_container_width=True)

        # Credits and resources
        st.divider()
        st.markdown(
            ":orange[**GitHub Repository:**]   \n"
            ":black[Explore our codebase and contribute to the future of tennis analysis on our [GitHub](https://github.com/feevemouad/TennisAnalysisProject) repository.]    \n"
            ":orange[**Contact Us:**]   \n"
            ":black[For inquiries or feedback, please reach out to us at [Gmail](mailto:mouad02aithammou@gmail.com).]"
        )
        return submitted, video_file

def main_page(submitted,video_file):
    """Main page layout.
    """
    ab = True
    with preprocessed_video.container():
        if submitted:
            if video_file is not None:
                st.divider()
                st.markdown("### :orange[**Input Video:**]")
                st.video(video_file)

            # if st.button("Process Video"):
            with st.spinner("Processing video... Stand up and strecth in the meantime 🙆‍♀️"):
                with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input_file, NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_file:
                    tmp_input_file.write(video_file.read())
                    tmp_input_file.flush()  # Ensure all data is written
                    os.fsync(tmp_input_file.fileno())  # Ensure all data is written to disk
                    tmp_input_file.seek(0)
                    process_video(tmp_input_file.name, tmp_output_file.name)
                    # Display processed video
                    st.divider()
                    st.markdown("### :orange[**Output Video:**]")
                    st.video(tmp_output_file.name[:-4] + "_h264.mp4")
                release_and_delete_file(tmp_input_file.name)
                release_and_delete_file(tmp_output_file.name)
                ab = False
    
            
    # If not submitted, chill here 🍹
        else:
            pass
    if ab:    
        with about.container():
            st.markdown("""
                **About**
                
                Welcome to Tennis Analysis System, your go-to platform for advanced tennis analysis using state-of-the-art technology. Our application integrates machine learning, computer vision, and deep learning techniques to provide comprehensive insights into tennis player performance. Whether you're a coach, player, or enthusiast, our interface offers intuitive tools to enhance your understanding of the game.

                """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image("video\img_source.png", caption="Input")
            with col2:
                st.image("video\img_output.png", caption="Output")
                
            st.markdown("""
                **Features**
                
                - **Player Detection**: Accurately identify players and tennis balls in videos using YOLOv8.
                - **Key Point Extraction**: Extract critical court keypoints using CNNs to understand player movements.
                - **Video Analysis**: Analyze player speed, ball shot speed, and shot counts with ease using CV2.
                - **Data-Driven Insights**: Leverage advanced analytics for a data-driven approach to feature development.

                **Get Started**
                Start analyzing tennis videos with ease! Simply upload your videos and explore a wealth of insights to improve your game.

                """)

# UI configurations
st.set_page_config(page_title="Tennis Analysis System",
                   page_icon="🎾",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Tennis Analysis System]")

# Placeholders for images and gallery
preprocessed_video = st.empty()
about = st.empty()

if __name__ == "__main__":
    submitted, video_file = configure_sidebar()
    main_page(submitted, video_file)