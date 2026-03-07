import cv2
import os
import sys

def extract_frame(video_path, timestamp_ms, output_name):
    """
    Extracts a single frame from a video at a specific millisecond.
    """
    if not os.path.exists(video_path):
        print(f"ERROR: Video file '{video_path}' not found.")
        return

    # Load the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    # Set the position to the specific millisecond
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    
    # Read the frame
    success, image = cap.read()
    if success:
        cv2.imwrite(output_name, image)
        print(f"SUCCESS: Saved frame at {timestamp_ms}ms to {output_name}")
    else:
        print(f"ERROR: Could not extract frame at {timestamp_ms}ms.")
    
    cap.release()

if __name__ == "__main__":
    # For testing purposes:
    # We use a placeholder name. In GSoC, you'll pass these as arguments.
    SAMPLE_VIDEO = "sample_news.mp4"
    TIMESTAMP = 5000  # 5 seconds in
    OUTPUT = "metaphor_check.jpg"
    
    print(f"Attempting to extract frame from {SAMPLE_VIDEO}...")
    extract_frame(SAMPLE_VIDEO, TIMESTAMP, OUTPUT)
