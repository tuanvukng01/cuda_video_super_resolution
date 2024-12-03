import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.esrgan_model import load_pretrained_model
from src.utils.video_processing import load_video, save_video
import numpy as np

import subprocess

def add_audio(input_audio_path, input_video_path, output_path):
    try:
        # Command to combine video and audio using FFmpeg
        ffmpeg_command = [
            "ffmpeg",
            "-i", input_video_path,
            "-i", input_audio_path,
            "-c:v", "copy",  # Copy the video codec
            "-c:a", "aac",  # Encode audio in AAC format
            "-strict", "experimental",  # Allow experimental features
            output_path
        ]

        # Run FFmpeg command
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error occurred: {result.stderr}")
        else:
            print(f"Final video with audio saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")



# Example usage


def main():
    # Define paths relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_video_path = os.path.join(project_root, "data", "input_video.mp4")
    output_video_path = os.path.join(project_root, "data", "output_video.mp4")

    # Load pre-trained model
    model = load_pretrained_model()
    model.eval()

    # Load input video
    frames = load_video(input_video_path)

    # Process video frames
    super_res_frames = []
    for frame in frames:
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            sr_frame = model(frame_tensor).squeeze(0).permute(1, 2, 0).numpy()

        # Normalize to 0-255 and convert to uint8
        sr_frame = (sr_frame - sr_frame.min()) / (sr_frame.max() - sr_frame.min()) * 255
        sr_frame = np.clip(sr_frame, 0, 255).astype(np.uint8)
        super_res_frames.append(sr_frame)

    # Save the super-resolution video
    save_video(super_res_frames, output_video_path)
    print(f"Super-resolution video saved to {output_video_path}")

if __name__ == "__main__":
    main()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_video_path = os.path.join(project_root, "data", "input_video.mp4")
    output_video_path = os.path.join(project_root, "data", "output_video.mp4")

    # Add audio from the input video to the output video
    final_output_path = os.path.join(project_root, "data", "output_with_audio.mp4")
    add_audio(input_video_path, output_video_path, final_output_path)
    print(f"Final video with audio saved to {final_output_path}")