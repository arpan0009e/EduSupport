
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# Model configuration
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_DIR = Path(__file__).resolve().parent

ADAPTER_PATH = BASE_DIR / "model" / "EduSupport_AI_Adapter"


def load_model():

    print("Loading EduSupport AI...")

    # Check adapter directory
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Adapter not found: {ADAPTER_PATH}"
        )

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16
    )

    # Attach fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH)
    )

    model.eval()

    print("EduSupport AI loaded successfully!")

    return model, tokenizer


def generate_response(model, tokenizer, branch, question):

    messages = [
        {
            "role": "system",
            "content": f"""
You are EduSupport AI, a friendly engineering student assistant.

Student Branch: {branch}

Instructions:
- Give technically correct answers.
- Use simple language.
- Keep answers concise.
- Avoid unnecessary details.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Prepare model prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Extract only generated text
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()
