
# **High-Performance Video Super-Resolution using CUDA**

Enhance video resolution with the power of deep learning! This project leverages a pre-trained **ESRGAN model** and **CUDA-optimized operations** to upscale video frames in real-time, demonstrating the practical application of AI and GPU acceleration.

---

## **Features**
- **CUDA-Optimized Processing**: Faster neural network computations with custom CUDA kernels.
- **State-of-the-Art Super-Resolution**: Utilizes the Enhanced Super-Resolution Generative Adversarial Network (**ESRGAN**) model to improve video quality.
- **Audio Retention**: Combines original audio with super-resolved video.
- **Benchmarking**: Compare processing speeds across **CPU**, **GPU**, and **CUDA** pipelines.

---

## **Directory Structure**
The project is organized as follows:
```plaintext
cuda_video_super_resolution/
├── data/                 # Input and output video files
│   ├── input_video.mp4   # Low-resolution input video
│   ├── output_video.mp4  # Super-resolved video (no audio)
│   └── output_with_audio.mp4  # Final super-resolved video with original audio
├── models/               # Pre-trained ESRGAN model implementation
│   ├── esrgan_model.py   # ESRGAN model definition and utilities
├── src/                  # Core source code
│   ├── cuda/             # CUDA kernel implementations
│   │   ├── matmul_kernel.cu
│   └── utils/            # Helper utilities for video processing
│       ├── video_processing.py
│   ├── main.py           # Main entry point
├── tests/                # Unit tests for models and utilities
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── .gitignore            # Ignored files
```

---

## **Requirements**
- **Python**: 3.8 or newer
- **Dependencies**:
  - `torch`: For PyTorch deep learning computations
  - `opencv-python`: For video processing
  - `numpy`: For matrix operations
  - `ffmpeg-python`: For merging video and audio
- **CUDA Toolkit**: Required for GPU acceleration (if running on compatible hardware)

---

## **Setup**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tuanvukng01/cuda_video_super_resolution.git
   cd cuda_video_super_resolution
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA Availability** (Optional):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## **Usage**

1. **Place Your Input Video**:
   - Add a low-resolution video to the `data/` directory and name it `input_video.mp4`.

2. **Run the Main Script**:
   ```bash
   python src/main.py
   ```

3. **View Output**:
   - Check `data/output_video.mp4` for the super-resolved video (without audio).
   - Check `data/output_with_audio.mp4` for the final video with audio.

---

## **Testing**
Run unit tests to ensure everything is working correctly:
```bash
python -m unittest discover -s tests
```

---

## **Performance**
### **Benchmark: CPU vs GPU vs CUDA**
| **Processing Pipeline** | **Estimated Runtime (300 frames)** |
|--------------------------|------------------------------------|
| **CPU**                 | ~50-100 minutes                  |
| **GPU (No CUDA)**        | ~10-20 minutes                   |
| **GPU (CUDA)**           | ~2-5 minutes                     |

---

## **Common Issues**
1. **OpenCV Fails to Read Input Video**:
   - Ensure `data/input_video.mp4` exists and is a valid format.
   - Test with `ffmpeg`:
     ```bash
     ffmpeg -i data/input_video.mp4
     ```

2. **No Audio in Final Video**:
   - Ensure the input video has an audio stream by checking:
     ```bash
     ffmpeg -i data/input_video.mp4
     ```

3. **CUDA Not Detected**:
   - Verify you have an NVIDIA GPU and CUDA Toolkit installed.

---

## **Future Work**
- Add support for real-time webcam super-resolution.
- Integrate live streaming with enhanced video quality.
- Experiment with other state-of-the-art super-resolution models.

---

## **Credits**
- **PyTorch**: Framework for deep learning.
- **OpenCV**: Video frame processing.
- **ffmpeg**: Video and audio handling.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.
