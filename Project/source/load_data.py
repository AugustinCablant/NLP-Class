import kagglehub
import os
import torch
from torch.utils.data import Dataset, DataLoader

books = ["01 Harry Potter and the Sorcerers Stone",
         "02 Harry Potter and the Chamber of Secrets",
         "03 Harry Potter and the Prisoner of Azkaban",
         "04 Harry Potter and the Goblet of Fire",
         "05 Harry Potter and the Order of the Phoenix",
         "06 Harry Potter and the Half-Blood Prince",
         "07 Harry Potter and the Deathly Hallows"]

def load():
    path = kagglehub.dataset_download("shubhammaindola/harry-potter-books")
    texts = {}
    TEXT = ""
    for book in books:
        file_path = os.path.join(path, f"{book}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        TEXT += text
        texts[book] = text
    return TEXT, texts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HarryDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size

        encoded_text = tokenizer.encode(text)

        self.inputs = []
        self.targets = []
        
        for i in range((len(encoded_text) - block_size)//10):
            i*=10
            # Input is a sequence of block_size characters
            input_seq = encoded_text[i:i+block_size]
            # Target is the next character after the input sequence
            target_seq = encoded_text[i+1:i+block_size+1]
            
            self.inputs.append(input_seq)
            self.targets.append(target_seq)
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx], dtype=torch.float), 
                torch.tensor(self.targets[idx], dtype=torch.float))
