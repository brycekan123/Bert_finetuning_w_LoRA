import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback, TrainingArguments, Trainer, set_seed
)
import wandb
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate

# Set random seed and device
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("ag_news")


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)


def compute_metrics(eval_pred):
    """Returns accuracy, precision, recall and f1 score for model training purposes"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# LoRA fine-tuning function
def train_eval_lora(model_name, r, lora_alpha, lora_dropout):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    full_train = dataset["train"].map(lambda x: tokenize(x, tokenizer), batched=True)
    full_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    full_test = dataset["test"].map(lambda x: tokenize(x, tokenizer), batched=True)
    full_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)

    # Configure LoRA 
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["query", "value"]
    )
    model = get_peft_model(base_model, peft_config).to(device)
    model.print_trainable_parameters()

    output_dir = f"./{model_name.replace('/', '_')}_lora_r{r}_alpha{lora_alpha}_drop{lora_dropout}"
    
    #Setting training parameters 
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        max_steps=10000,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"{model_name.split('/')[-1]}_r{r}_alpha{lora_alpha}_drop{lora_dropout}" 
    )
    #Implemented early stopping for efficiency
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train,
        eval_dataset=full_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\nTraining {model_name} with LoRA params: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    trainer.train()

    # Save LoRA adapter weights and config
    model.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")
    return metrics["eval_accuracy"], output_dir

# Evaluate base model accuracy
def eval_base_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    full_test = dataset["test"].map(lambda x: tokenize(x, tokenizer), batched=True)
    full_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    #In this dataset, we have 4 labels: World, Sports, Business, Sci/Tech
    base_model_plain = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)

    base_trainer = Trainer(
        model=base_model_plain,
        args=TrainingArguments(
            output_dir=f"./base_eval_{model_name.replace('/', '_')}",
            per_device_eval_batch_size=32,
            fp16=torch.cuda.is_available()
        ),
        eval_dataset=full_test,
        compute_metrics=compute_metrics,
    )
    base_metrics = base_trainer.evaluate()
    return base_metrics["eval_accuracy"]

# === Main Setup ===
#Chose this configuration to achieve max accuracy within reasonable amount of training time
#gives ability to test different ranks, alpha and dropout values
lora_param_grid = [{"r": 16, "lora_alpha": 64, "lora_dropout": 0.1}]
#The 2 models I chose
model_names = ["google/bert_uncased_L-2_H-128_A-2", "prajjwal1/bert-tiny"]

results = []
saved_models = {}

for model_name in model_names:
    print(f"\nEvaluating base model: {model_name}")
    base_acc = eval_base_model(model_name)
    print(f"Base accuracy for {model_name}: {base_acc:.4f}")
    results.append({
        "model": model_name,
        "r": None,
        "alpha": None,
        "dropout": None,
        "accuracy": base_acc,
        "note": "base"
    })

    for params in lora_param_grid:
        acc, output_dir = train_eval_lora(model_name, params["r"], params["lora_alpha"], params["lora_dropout"])
        results.append({
            "model": model_name,
            "r": params["r"],
            "alpha": params["lora_alpha"],
            "dropout": params["lora_dropout"],
            "accuracy": acc,
            "note": "lora"
        })
        saved_models[model_name] = output_dir

#Get table of results
print("\n=== Final Results ===")
print(tabulate(results, headers="keys", floatfmt=".4f"))

#How well does fine tuned model actually perform on data? See below!

# Define models and LoRA output directory
chosen_model = "google/bert_uncased_L-2_H-128_A-2"
chosen_output_dir = "./google_bert_uncased_L-2_H-128_A-2_lora_r16_alpha64_drop0.1"

print(f"\nLoaded base model: {chosen_model}")
print(f"Using fine-tuned LoRA adapter from: {chosen_output_dir}")

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(chosen_model)

# Load base model without LoRA
base_model_plain = AutoModelForSequenceClassification.from_pretrained(chosen_model, num_labels=4).to(device)
base_model_plain.eval()

# Load base model then add LoRA adapter weights
base_model_for_lora = AutoModelForSequenceClassification.from_pretrained(chosen_model, num_labels=4).to(device)
model = PeftModel.from_pretrained(base_model_for_lora, chosen_output_dir).to(device)
model.eval()

label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Select 10 random samples from test set for inference
random.seed(100)
sample_indices = random.sample(range(len(dataset["test"])), 10)
samples = [dataset["test"][i] for i in sample_indices]
texts = [s['text'] for s in samples]
true_labels = [s['label'] for s in samples]

def predict(texts, model, tokenizer):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    return preds.cpu().tolist(), probs.cpu().tolist()

#Comparing predictions of base model and fine tuned model
base_preds, base_probs = predict(texts, base_model_plain, tokenizer)
lora_preds, lora_probs = predict(texts, model, tokenizer)

print("\n=== Predictions on Random Sample Texts ===\n")
for i, text in enumerate(texts):
    print(f"Text: {text}\n")
    print(f"True label: {label_map[true_labels[i]]}")
    print(f"Base model prediction: {label_map[base_preds[i]]} (confidence: {max(base_probs[i]):.2f})")
    print(f"LoRA fine-tuned prediction: {label_map[lora_preds[i]]} (confidence: {max(lora_probs[i]):.2f})")
    print("-" * 80)
