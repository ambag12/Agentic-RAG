import os
import requests
from gpt4all import GPT4All

# 1. Define model directory and filename
model_dir = r"C:\Agentic-Rag-by-Model"
os.makedirs(model_dir, exist_ok=True)

model_filename = "ggml-model-q4_0.gguf"
model_path = os.path.join(model_dir, model_filename)

# 2. Download the model if it doesn't exist
model_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF/resolve/main/ggml-model-q4_0.gguf"

if not os.path.exists(model_path):
    print(f"Downloading model to {model_path}...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()  # Raises an error for unsuccessful status codes
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")
else:
    print("Model already exists.")

# 3. Load the model without the unsupported 'backend' argument
print("Loading model...")
model = GPT4All(
    model_name=model_filename,
    model_path=model_dir,
    verbose=True
)

# 4. Generate text
print("Generating response...")
response = model.generate(
    prompt="Hello, how are you?",
    max_tokens=50,
    temp=0.7
)

print("Response:")
print(response)
