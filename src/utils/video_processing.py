import cv2
import numpy as np

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:  # Handle empty frames
            print("Warning: Skipping an empty frame.")
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Failed to read frames from video: {video_path}")
    return frames


def save_video(frames, output_path, fps=30):
    if len(frames) == 0:
        raise ValueError("No frames to save in the video!")

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        # Convert to uint8 if necessary
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255)  # Ensure values are within range
            frame = frame.astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()