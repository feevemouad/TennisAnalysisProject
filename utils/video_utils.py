import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(video_frames, output_path):
    if len(video_frames) == 0:
        return
    height, width, layers = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    video = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    for frame in video_frames:
        video.write(frame)
    video.release()
