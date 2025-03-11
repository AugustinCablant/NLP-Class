# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import kagglehub
import os
import pandas as pd
from transformers import GPT2Tokenizer
from source.load_data import load, HarryDataset
from tokenize import Tokenizer

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 32


# ------------

torch.manual_seed(2025)

# import data
tokenizer = Tokenizer()
TEXT, texts = load()
dataloader = DataLoader(HarryDataset(texts, tokenizer), batch_size=batch_size, shuffle=True)


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



model =

def train(loader, lr=0.001):
    # Hyperparameters
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    total_loss = 0
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
    
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Print stats every 10 iterations
        if (i + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f"Iteration {i+1}, Average Loss: {avg_loss:.4f}")
            total_loss = 0  # Reset total loss
    print("Training completed!")


def generate_text(model, input_text, limit=5000):
    model.eval()
    text = input_text
    for i in range(limit):
        inputs = tokenizer.encode(text)
        inputs = torch.tensor(inputs, dtype=torch.float).to(device).unsqueeze(0)
        outputs = model(inputs)
        outputs = outputs[:, :, :].round().cpu().detach().numpy()[:, -1, :]
        output_text = tokenizer.decode(outputs)
        text += output_text
    print(text)


input = '''Ron never imagined that Harry would '''
generate_text(model, input)


# generate from the model
prompt = torch.tensor(encode(['\n']))
context = torch.ones((1,1), dtype=torch.long, device=device)*prompt
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))