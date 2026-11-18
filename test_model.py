
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# Model paths
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./model/EduSupport_AI_Adapter"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Qwen base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Attaching EduSupport AI adapter...")

model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

model.eval()

print("\nEduSupport AI loaded successfully!")

print("GPU memory allocated:",
      round(torch.cuda.memory_allocated(0) / 1024**3, 2),
      "GB")

# Test question
messages = [
    {
        "role": "system",
        "content": (
            "You are EduSupport AI. "
            "The student's branch is AI & ML. "
            "Give simple and concise engineering explanations."
        )
    },
    {
        "role": "user",
        "content": "What is machine learning?"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

print("\nGenerating response...")

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nEduSupport AI Response:")
print(response)
