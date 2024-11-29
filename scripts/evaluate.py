from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss)

model_path = "./fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

test_text = "Maschinelles Lernen ist eine Methode der künstlichen Intelligenz."
perplexity = calculate_perplexity(model, tokenizer, test_text)
print("Perplexity:", perplexity.item())