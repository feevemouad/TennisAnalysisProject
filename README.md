# `Tennis Analysis System`

### Overview

The Tennis Analysis System leverages advanced computer vision techniques to analyze tennis matches. By detecting keypoints of the court, players, and the ball, the system provides detailed insights into player movements and ball dynamics.

### Demo
https://github.com/feevemouad/TennisAnalysisProject/assets/105555280/959440a4-41e5-43c2-89d2-c4f5241b78bd

### Features

- **Keypoint Detection**: Uses transfer learning on the MobileNetV3 model to accurately detect keypoints of the tennis court ([ dataset used ](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link)).
- **Object Detection**: Finetuned YOLOv8 on a custom dataset from [Roboflow](https://universe.roboflow.com/tennistracker-dogbm/tennis-tracker-duufq/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true) to detect players and the ball with high precision.
- **Mini Map Representation**: A simplified representation of the match showing the court, players as dots, and the ball moving between players.
- **Streamlit Interface**: An interactive web interface built with Streamlit for easy analysis and visualization of the match data.
- **Video Analysis**: Analyze player speed, ball shot speed, and shot counts using OpenCV. (To Be added)
