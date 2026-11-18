
from model_loader import load_model, generate_response

# Load model
model, tokenizer = load_model()

# Test question
answer = generate_response(
    model=model,
    tokenizer=tokenizer,
    branch="AI & ML",
    question="What is deep learning?"
)

print("\nEduSupport AI:")
print(answer)
