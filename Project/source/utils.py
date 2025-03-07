import pandas as pd 
import os

def create_dataframe_train():
    types = ['neg', 'pos', 'unsup']
    rates = []
    IDs = []
    sentences = []
    data = []
    for type in types:
        folder = f"data/train/{type}"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            data.append({"ID": filename[0], 
                         "Sentence": content,
                         "Rate": filename[2],
                         "Type": type})
    df = pd.DataFrame(data)
    return df 