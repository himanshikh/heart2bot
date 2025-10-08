import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Use tiny model
model_name = "sshleifer/tiny-gpt2"  # very small model, CPU-friendly
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input
inp = tok("The user said: I feel", return_tensors="pt")

# Forward pass only (no generate)
with torch.no_grad():
    output = model(**inp)

print("Forward pass OK, output shape:", output.logits.shape)
