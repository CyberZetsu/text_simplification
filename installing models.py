# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("nvidia/Llama-3_3-Nemotron-Super-49B-v1", trust_remote_code=True)