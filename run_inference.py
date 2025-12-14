# run_inference.py
import torch
import time

print("--- Starting Red Hen Inference Job ---")

# 1. Check for GPU (The critical check for HPC)
if torch.cuda.is_available():
    print(f"SUCCESS: Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected. Running on CPU (Not recommended for Production).")

# 2. Simulation of a heavy task (Loading a model)
print("Loading Llama-3-8B (Simulation)...")
time.sleep(2) # Simulate load time
print("Model loaded. Starting batch processing...")

# 3. Simulate processing
for i in range(5):
    print(f"Processing Batch {i+1}/5...")
    time.sleep(1)

print("--- Job Complete. Output saved to /results ---")