import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class QuoraDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(row['question1'], row['question2'],
                                padding='max_length', max_length=self.max_length, truncation=True)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        label = torch.tensor(row['is_duplicate'])
        random_score = torch.tensor(row['random_score'], dtype=torch.float)
        return inputs, label, random_score
    
    @staticmethod
    def load_data(path,size=None):
        data = pd.read_csv(path)
        # Assuming the dataset columns are ['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
        data = data.dropna()  # Removing rows with null values
        data = data.sample(size) if size else data
        data.drop(columns=data.columns[:3],inplace=True)
        return data

    @staticmethod
    def split_data(data, test_size=0.1, val_size=0.1):
        train_val, test = train_test_split(data, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        return train, val, test

    @staticmethod
    def tokenize_questions(question1, question2, max_length=128):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer(question1, question2, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')