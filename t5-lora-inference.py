import torch
from transformers import T5Tokenizer
from peft import PeftModel, PeftConfig

# Check for MPS availability (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Path to your saved model
model_path = "./t5-paraphrase-lora"

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Load the PEFT configuration
config = PeftConfig.from_pretrained(model_path)
# Load the base model specified in the config
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
# Load the LoRA weights on top of the base model
model = PeftModel.from_pretrained(model, model_path)
model = model.to(device)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

def generate_paraphrase(text):
    """Generate a paraphrase for the given text."""
    input_text = f"paraphrase : {text}"
    print(f"Input: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Use keyword arguments for generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            num_beams=4,  # Use beam search for better quality
            no_repeat_ngram_size=2,  # Avoid repeating phrases
            temperature=0.7  # Add some randomness for diversity
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Example questions to test
test_questions = [
    "What is the capital of France?",
    "How do I reset my password?",
    "Carol's current balance is $50 but she's trying to make a $75 purchase with her cash card. How will the CIV system respond?",
    "Can you explain the difference between machine learning and deep learning?",
    "What are the steps to troubleshoot a network connectivity issue?"
]

print("\nTesting model with example questions:")
print("=" * 80)
for question in test_questions:
    paraphrase = generate_paraphrase(question)
    print(f"Original: {question}")
    print(f"Paraphrase: {paraphrase}")
    print("-" * 80)

# Interactive mode
print("\nInteractive Mode: Enter questions to paraphrase (type 'exit' to quit)")
while True:
    user_input = input("\nEnter text to paraphrase: ")
    if user_input.lower() == 'exit':
        break
    
    paraphrase = generate_paraphrase(user_input)
    print(f"Paraphrase: {paraphrase}")
