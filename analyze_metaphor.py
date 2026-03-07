import torch
import clip
from PIL import Image
import cv2
import os
import numpy as np

def run_clip_analysis(image_path, text_descriptions):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare the image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    
    # Prepare the text prompts (the metaphors)
    text = clip.tokenize(text_descriptions).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs

if __name__ == "__main__":
    IMG_PATH = "metaphor_check.jpg"
    METAPHORS = ["a calm landscape", "a chaotic financial crash", "a fast race"]
    
    if os.path.exists(IMG_PATH):
        print(f"Analyzing {IMG_PATH} against metaphors...")
        results = run_clip_analysis(IMG_PATH, METAPHORS)
        
        for i, metaphor in enumerate(METAPHORS):
            print(f"Confidence for '{metaphor}': {results[0][i]:.4f}")
    else:
        print("Error: Image not found. Run extract_frames.py first.")
