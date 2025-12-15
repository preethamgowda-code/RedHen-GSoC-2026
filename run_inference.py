# run_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Setup - Select a model
# We use 'gpt2' for this test because it's small, fast, and public.
# In the real GSoC project, we will swap this for 'meta-llama/Meta-Llama-3-8B'
MODEL_NAME = "gpt2"

print(f"--- 🚀 Starting NLP Pipeline Test with {MODEL_NAME} ---")

try:
    # 2. Load the Tokenizer (The Translator)
    # This converts "Hello" into numbers like [15496]
    print(f"Downloading/Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. The Input Data (Frame Blending Example)
    text_input = "The stock market dived this morning."
    print(f"\n📥 Input Text: '{text_input}'")

    # 4. Tokenization (The Core NLP Task)
    # return_tensors="pt" means "Give me PyTorch tensors" (which GPUs need)
    inputs = tokenizer(text_input, return_tensors="pt")
    
    print("\n🔢 Tokenized Output (Tensor):")
    print(inputs['input_ids'])
    
    # 5. Proof of Understanding (Decoding back)
    # We verify that the numbers actually represent the words
    decoded_text = tokenizer.decode(inputs['input_ids'][0])
    print(f"\n✅ Decoded Check: '{decoded_text}'")
    
    print("\n--- Success: Environment is ready for LLM Inference ---")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("Tip: Make sure you have an internet connection to download the model config.")