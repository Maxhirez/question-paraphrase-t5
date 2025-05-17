import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from evaluate import load
import nltk
nltk.download('punkt')

# Check for MPS availability (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets from GLUE: MRPC and QQP
print("Loading datasets...")
mrpc_dataset = load_dataset("glue", "mrpc")
qqp_dataset = load_dataset("glue", "qqp")

# Load T5 model and tokenizer
model_name = "t5-base"
print(f"Loading {model_name}...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)  # Move model to MPS

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,  # rank of the update matrices
    lora_alpha=32,  # scaling factor
    lora_dropout=0.1,
    # Target only attention layers for efficiency
    target_modules=["q", "v"],
    bias="none",
)

# Apply LoRA to the model
print("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows what % of parameters are being trained

# Process MRPC dataset
def process_mrpc(dataset):
    # Filter to only positive examples (label == 1)
    filtered_dataset = dataset.filter(lambda x: x["label"] == 1)
    
    # Process the dataset
    processed_dataset = filtered_dataset.map(
        lambda examples: {
            "input_ids": tokenizer(["paraphrase: " + sent for sent in examples["sentence1"]], 
                                  padding="max_length", truncation=True, max_length=128).input_ids,
            "attention_mask": tokenizer(["paraphrase: " + sent for sent in examples["sentence1"]], 
                                      padding="max_length", truncation=True, max_length=128).attention_mask,
            "labels": tokenizer(examples["sentence2"], 
                               padding="max_length", truncation=True, max_length=128).input_ids
        },
        batched=True,
        remove_columns=filtered_dataset.column_names
    )
    
    return processed_dataset

# Process QQP dataset
def process_qqp(dataset):
    # Filter to only positive examples (label == 1)
    filtered_dataset = dataset.filter(lambda x: x["label"] == 1)
    
    # Process the dataset
    processed_dataset = filtered_dataset.map(
        lambda examples: {
            "input_ids": tokenizer(["paraphrase: " + q for q in examples["question1"]], 
                                  padding="max_length", truncation=True, max_length=128).input_ids,
            "attention_mask": tokenizer(["paraphrase: " + q for q in examples["question1"]], 
                                      padding="max_length", truncation=True, max_length=128).attention_mask,
            "labels": tokenizer(examples["question2"], 
                               padding="max_length", truncation=True, max_length=128).input_ids
        },
        batched=True,
        remove_columns=filtered_dataset.column_names
    )
    
    return processed_dataset

# Process train and validation datasets
print("Processing MRPC dataset...")
mrpc_train = process_mrpc(mrpc_dataset["train"])
mrpc_val = process_mrpc(mrpc_dataset["validation"])

print("Processing QQP dataset...")
qqp_train = process_qqp(qqp_dataset["train"])
# Use a subset of QQP for validation to keep validation fast
qqp_val = process_qqp(qqp_dataset["validation"].select(range(min(len(qqp_dataset["validation"]), 1000))))

# Combine datasets
print("Combining datasets...")
train_dataset = concatenate_datasets([mrpc_train, qqp_train])
validation_dataset = concatenate_datasets([mrpc_val, qqp_val])

print(f"Combined train dataset size: {len(train_dataset)}")
print(f"Combined validation dataset size: {len(validation_dataset)}")

# Shuffle the training dataset
train_dataset = train_dataset.shuffle(seed=42)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

# Metric for evaluation
rouge = load("rouge")

def compute_metrics(eval_preds):
    # preds, labels = eval_preds
    
    # # Ensure preds is 2D (batch_size, seq_length)
    # if isinstance(preds, list):
    #     if isinstance(preds[0], list):
    #         preds = preds[0]  # Take first element if it's a list of lists
    
    # # Decode predictions and labels
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # # Handle potential issues with prediction format
    # try:
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # except TypeError:
    #     # If preds is not the right format, try to fix it
    #     preds_fixed = np.array(preds).reshape(-1, np.array(preds).shape[-1])
    #     decoded_preds = tokenizer.batch_decode(preds_fixed, skip_special_tokens=True)
    
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # # Rouge expects a newline after each sentence
    # decoded_preds = [pred.strip() for pred in decoded_preds]
    # decoded_labels = [label.strip() for label in decoded_labels]
    
    # # Compute ROUGE scores
    # result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # # Extract ROUGE scores
    # result = {k: round(v * 100, 4) for k, v in result.items()}
    
    # # Add mean generated length
    # prediction_lens = [len(tokenizer.encode(pred)) for pred in decoded_preds]
    # result["gen_len"] = np.mean(prediction_lens)
    
    # return result
    return {"rouge1":0.0}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=False,  # Change to False since MPS doesn't support fp16
    use_mps_device=torch.backends.mps.is_available(),
    report_to="none",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
model.save_pretrained("./t5-paraphrase-lora")
tokenizer.save_pretrained("./t5-paraphrase-lora")

# Test the model on some examples
def generate_paraphrase(text):
    input_ids = tokenizer(f"paraphrase: {text}", return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example questions to test
test_questions = [
    "What is the capital of France?",
    "How do I reset my password?",
    "Carol's current balance is $50 but she's trying to make a $75 purchase with her cash card. How will the CIV system respond?"
]

print("\nTesting model with example questions:")
print("Original vs Paraphrased:")
for question in test_questions:
    paraphrase = generate_paraphrase(question)
    print(f"Original: {question}")
    print(f"Paraphrase: {paraphrase}")
    print("-" * 50)
