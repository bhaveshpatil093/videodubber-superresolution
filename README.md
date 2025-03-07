# Videodubber Superresolution Enhancement

This repository contains the solution for enhancing the resolution of lip-synced videos generated by MuseTalk. The script applies superresolution using either GFPGAN or CodeFormer to improve the generated subframe resolution.

---

## Features
- Enhance lip-synced videos by applying superresolution to the generated subframe only.
- Supports both GFPGAN and CodeFormer for superresolution.
- CLI interface for seamless usage.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/videodubber-superresolution.git
   cd videodubber-superresolution

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

Download the pretrained models:

1. [GFPGAN model](https://github.com/TencentARC/GFPGAN)
2. [CodeFormer model](https://github.com/sczhou/CodeFormer)

Save the downloaded models in the models/ directory.

## Usage

### Command

    ```bash
    python x.py --superres [GFPGAN/CodeFormer] -iv <input_video_path> -ia <input_audio_path> -o <output_video_path>

### Example

    ```bash
    python x.py --superres GFPGAN -iv test_data/input.mp4 -ia test_data/input.mp3 -o outputs/output_example.mp4

### Inputs

1. --superres: Choose between GFPGAN or CodeFormer for superresolution.
2. -iv: Path to the input video file.
3. -ia: Path to the input audio file.
4. -o: Path for saving the output video file.

### Outputs

The output video will be saved in the specified path, with improved resolution for the generated subframe.

## Testing

1. Use the provided test_data/input.mp4 and test_data/input.mp3 files for testing.
2. Run the command:
    ```bash
    python x.py --superres GFPGAN -iv test_data/input.mp4 -ia test_data/input.mp3 -o outputs/output_example.mp4
3. Check the outputs/ folder for the resulting enhanced video.

## Requirements

1. Python 3.7+
2. OpenCV
3. Torch
4. Numpy
5. MoviePy
6. FFmpeg
7. GFPGAN and CodeFormer pretrained models

Install all the dependencies with:
    ```bash
    pip install -r requirements.txt

## License

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
