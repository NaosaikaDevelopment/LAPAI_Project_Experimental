import os
from transformers import AutoTokenizer
current_dir = os.path.dirname(os.path.abspath(__file__))

save_path = os.path.join(current_dir, "tokenizer_files")

print(f"Downloading to: {save_path}...")

model_id = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=save_path
)

