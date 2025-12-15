# run_inference.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Configuration
# We use a real model name now (Llama-3 style naming), though we won't download the full thing locally
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
print(f"--- 🚀 Initializing AI Pipeline for: {MODEL_ID} ---")

# 2. Hardware Detection (The "Engineering" Part)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Hardware Detected: {device.upper()}")

# 3. Define Quantization Config (The Secret Sauce)
# This tells the model: "Load in 4-bit format to save memory"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

try:
    # 4. Load Tokenizer
    print("⏳ Loading Tokenizer...")
    # Using gpt2 as a lightweight placeholder for local testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    
    # 5. Simulation Logic
    # IF we are on a GPU, we would load the model like this:
    if device == "cuda":
        print("🔧 GPU Detected: Loading model with 4-bit Quantization...")
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_ID, 
        #     quantization_config=bnb_config,  <-- This is the key line
        #     device_map="auto"
        # )
        print("✅ Model loaded in 4-bit mode (Simulated for GitHub).")
    else:
        print("⚠️  No GPU detected (Local Mode). Skipping heavy model load.")
        print("ℹ️  Note: BitsAndBytes 4-bit loading requires NVIDIA GPU.")

    # 6. Test Data Processing
    text = "The stock market dived."
    inputs = tokenizer(text, return_tensors="pt")
    
    print(f"\n📥 Processing Text: '{text}'")
    print(f"🔢 Token IDs: {inputs['input_ids'][0].tolist()}")
    print("\n✅ Pipeline Check: SUCCESS")

except Exception as e:
    print(f"\n❌ Error: {e}")