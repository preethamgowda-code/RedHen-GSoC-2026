import os
import subprocess

def run_pipeline(video_file, timestamp, metaphors):
    print(f"--- STARTING RED HEN BLENDER PIPELINE ---")
    
    # 1. Extract the frame
    print(f"STEP 1: Extracting frame from {video_file} at {timestamp}ms...")
    extract_cmd = f"python3 extract_frames.py" 
    # (Note: In a real run, we'd pass args, but for now we run your existing script)
    os.system(extract_cmd)
    
    # 2. Analyze the metaphor
    print(f"STEP 2: Analyzing visual metaphors using CLIP...")
    analyze_cmd = f"python3 analyze_metaphor.py"
    os.system(analyze_cmd)
    
    print(f"--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline("sample_news.mp4", 5000, ["landscape", "crash", "race"])
