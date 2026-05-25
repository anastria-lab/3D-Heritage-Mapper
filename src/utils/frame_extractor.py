import cv2
import os


def extract_frames(video_path, output_folder, frame_skip=10):
    """
    Extracts frames from a video file.
    :param video_path: Path to the input video.
    :param output_folder: Where to save the images.
    :param frame_skip: Saves every Nth frame (e.g., 10 means 1 frame saved every 10 frames).
    """
    print(f"Opening video: {video_path}")

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open the video. Check the file path.")
        return

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    saved_count = 0

    print("Extracting frames... (This might take a minute)")

    while True:
        success, frame = video.read()

        if not success:
            break

        # Only save every Nth frame
        if frame_count % frame_skip == 0:
            # Create a file name with leading zeros (e.g., frame_0001.jpg)
            file_name = f"frame_{saved_count:04d}.jpg"
            save_path = os.path.join(output_folder, file_name)

            # Save the image
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    # Cleanup
    video.release()
    print(f"\nDone! Extracted {saved_count} frames to {output_folder}")


if __name__ == "__main__":
    # Example Usage:
    # Change these paths to test it with a real video!
    sample_video = "data/raw/my_test_video.mp4"
    destination = "data/processed/statue_frames"
