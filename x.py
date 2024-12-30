import argparse
import os
import cv2
from musetalk import generate_lipsynced_video  # Assuming musetalk is installed/imported
from superres_utils import apply_superresolution  # Custom utility script

def main():
    # Argument parser for CLI usage
    parser = argparse.ArgumentParser(description="Enhance video quality using GFPGAN or CodeFormer.")
    parser.add_argument('--superres', choices=['GFPGAN', 'CodeFormer'], required=True,
                        help="Superresolution model to use")
    parser.add_argument('-iv', '--input_video', required=True, help="Path to input video")
    parser.add_argument('-ia', '--input_audio', required=True, help="Path to input audio")
    parser.add_argument('-o', '--output', required=True, help="Path to output video")

    args = parser.parse_args()

    # Step 1: Generate lip-synced video
    print("Generating lip-synced video...")
    temp_video_path = "temp_video.mp4"  # Temporary output for lip-synced video
    generate_lipsynced_video(args.input_video, args.input_audio, temp_video_path)

    # Step 2: Apply superresolution
    print(f"Applying {args.superres} superresolution on the generated subframes...")
    enhanced_video_path = apply_superresolution(
        input_video=temp_video_path,
        model=args.superres,
        output_video=args.output
    )

    # Step 3: Cleanup temporary files
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    print(f"Enhanced video saved at: {enhanced_video_path}")


if __name__ == "__main__":
    main()
