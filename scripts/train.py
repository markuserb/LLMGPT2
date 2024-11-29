from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Datensatz laden
train_dataset_path = "/Users/markuserb/dev/Python/LLM/data/dataset.txt"  # Pfad zum Datensatz
eval_dataset_path = "/Users/markuserb/dev/Python/LLM/data/eval_dataset.txt"

# Lade den Datensatz für Training und Evaluation
train_dataset = load_dataset("text", data_files={"train": train_dataset_path})["train"]
eval_dataset = load_dataset("text", data_files={"eval": eval_dataset_path})["eval"]

# Tokenizer und Modell laden
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenizer anpassen (alle Eingabesequenzen die selbe Länge)
tokenizer.pad_token = tokenizer.eos_token

# 3. Datensatz tokenisieren
def tokenize_function(examples):
    # Tokenisiere den Text und setze die Labels gleich wie die input_ids
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Labels sind identisch mit input_ids
    return tokenized_inputs

# Tokenisierung für Trainings- und Evaluierungsdatensatz
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Trainingsparameter festlegen
training_args = TrainingArguments(
    output_dir="./results",              # Ordner für Speicherung von Modell und Logs
    evaluation_strategy="epoch",         # Evaluiert nach jeder Epoche
    learning_rate=5e-5,
    per_device_train_batch_size=4,       # Trainings-Batch-Größe festlegen
    per_device_eval_batch_size=8,        # Evaluierungs-Batch-Größe festlegen
    num_train_epochs=3,                  # 3 Trainings-Epochen 
    save_total_limit=1,                  # Speichert nur das neueste Modell
    logging_dir="./logs",                # Logs speichern
    logging_steps=10                     # Alle 10 Schritte wird geloggt
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,   
    eval_dataset=tokenized_eval,     
)

# Training starten
trainer.train()

# Modell speichern
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Modell evaluieren
results = trainer.evaluate()
print(results)