from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Modell und Tokenizer laden
model_path = "./fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Starttext (Prompt) für die Generierung
prompt = "Das Wetter ist heute wirklich angenehm und"

# Tokenisierung des Starttextes
inputs = tokenizer(prompt, return_tensors="pt")

# Attention Mask erstellen: Setzt alle Eingabepositionen auf 1 (keine Padding-Tokens)
attention_mask = torch.ones(inputs["input_ids"].shape, dtype=torch.long)

# Textgenerierung mit zusätzlichen Parametern für Sampling und Kontrolle
generated_output = model.generate(
    inputs["input_ids"], 
    attention_mask=attention_mask,  # attention_mask hinzufügen
    max_length=50,  # Maximale Länge der generierten Textfolge
    num_return_sequences=1,  # Anzahl der zu generierenden Texte
    no_repeat_ngram_size=2,  # Verhindert die Wiederholung von n-Grammen
    temperature=0.7,  # Steuerung der Kreativität (0.7 ist ein guter Wert)
    top_k=50,  # Top-k Sampling
    top_p=0.95,  # Nucleus Sampling (kontrolliert die Wahrscheinlichkeit)
    do_sample=True,  # Aktiviert das Sampling (für variierende Ausgaben)
    pad_token_id=tokenizer.eos_token_id  # Pad Token ID auf EOS-Token setzen
)

# Den generierten Text dekodieren
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

print("Generierter Text:")
print(generated_text)