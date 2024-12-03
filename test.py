import cv2

video_path = "data/input_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video file: {video_path}")
else:
    print("Video file opened successfully!")