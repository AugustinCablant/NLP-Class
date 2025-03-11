import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import kagglehub
import os
import pandas as pd
from transformers import T5Tokenizer
from source.load_data import load, HarryDataset
from transformers import T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader
import re
from tqdm import tqdm

batch_size = 16
epochs = 100
eval_interval = 500
learning_rate = 5e-5
max_length = 512
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.cuda.empty_cache()

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Import our dataset
TEXT, texts = load()
dataset = re.split(r'(?<=[.!?])\s+', TEXT)

# Function to preprocess a sentence
def preprocess_sentence(sentence):
    input_text = f"paraphrase: {sentence}"  
    output_text = sentence  

    # Tokenize input and output
    input_tokens = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    output_tokens = tokenizer(output_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    
    # Return a dictionary with tokenized inputs and outputs
    return {
        "input_ids": input_tokens.input_ids.squeeze(0),
        "attention_mask": input_tokens.attention_mask.squeeze(0),
        "labels": output_tokens.input_ids.squeeze(0)  # Using labels for the target text
    }

# Preprocess the dataset
processed_data = [preprocess_sentence(sentence) for sentence in dataset]

# Initialize the model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Prepare the data for DataLoader
train_dataloader = DataLoader(processed_data, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Move the model to GPU if available
model.to(device)

# Training loop
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_dataloader)}")

# Function to generate paraphrases or transformed sentences
def generate_paraphrase(input_sentence):
    # Tokenize input sentence
    input_tokens = tokenizer(input_sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = input_tokens.input_ids.to(device)

    # Generate the output
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

# Text generation
with open('prompt.txt', 'r') as file:
    input_sentence = file.read()

generated_sentence = generate_paraphrase(input_sentence)
print(generated_sentence)