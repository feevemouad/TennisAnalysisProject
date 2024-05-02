import cv2
import numpy as np
import tensorflow as tf

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))
        image_batch = np.expand_dims(image_resized, axis=0)
        outputs = self.model.predict(image_batch)
        keypoints = outputs[0]
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 222), 2)
            cv2.circle(image, (x, y), 3, (0, 0, 222), -1)
        return image
    
