from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack"
]

# 1️⃣ Load ONLINE dataset
logger.info("Loading civil_comments dataset...")
dataset = load_dataset("civil_comments", split="train")

# Reduce dataset size (IMPORTANT for laptops)
dataset = dataset.shuffle(seed=42).select(range(20000))
logger.info(f"Selected {len(dataset)} samples for training")

# Split into train and validation
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']
logger.info(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

# 2️⃣ Convert float labels → FLOAT multi-label tensor
def process_labels(example):
    example["labels"] = np.array(
        [1.0 if example[label] >= 0.5 else 0.0 for label in LABELS],
        dtype=np.float32
    )
    return example

logger.info("Processing labels...")
train_dataset = train_dataset.map(process_labels)
val_dataset = val_dataset.map(process_labels)

# 3️⃣ Tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

logger.info("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 4️⃣ Set dataset format (CRITICAL FIX)
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
    output_all_columns=False
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
    output_all_columns=False
)

# 5️⃣ Load BERT (multi-label)
logger.info("Loading BERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

# 6️⃣ Metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).float().numpy()
    
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

# 7️⃣ Training arguments (Mac-friendly)
logger.info("Setting up training configuration...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=False
)

# 8️⃣ Trainer
logger.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 9️⃣ Train
logger.info("Starting training...")
trainer.train()

# 🔟 Evaluate
logger.info("Evaluating model...")
eval_results = trainer.evaluate()
logger.info(f"Evaluation results: {eval_results}")

# ⓫ Save model
logger.info("Saving model...")
model.save_pretrained("bert_toxic_model")
tokenizer.save_pretrained("bert_toxic_model")

print("\n" + "="*50)
print("✅ Multi-label BERT model trained successfully")
print(f"📊 Final F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}")
print(f"📊 Final Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
print("="*50)

