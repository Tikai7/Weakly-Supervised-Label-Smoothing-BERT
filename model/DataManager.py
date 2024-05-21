import pyterrier as pt
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
        # Récupérer une ligne du dataframe
        row = self.data.iloc[idx]
        # Tokenizer les questions 1 et 2 
        inputs = self.tokenizer(row['question1'], row['question2'],
                                padding='max_length', max_length=self.max_length, truncation=True)
        inputs = {key: torch.tensor(val) for key, val in inputs.items()}
        # Récupérer le label et le score
        label = torch.tensor(row['is_duplicate'])
        score = torch.tensor(row['score'], dtype=torch.float)
        return inputs, label, score

    @staticmethod
    def remove_overlapping_questions(train_df, other_df, question="question1"):
        # permet de retirer les questions qui sont déjà dans le train_df (question 1 ou 2)
        print("Shape before : ", other_df.shape)
        train_df['combined'] = train_df[question] 
        other_df['combined'] = other_df[question] 
        train_questions = set(train_df['combined'].unique())
        filtered_other_df = other_df[~other_df['combined'].isin(train_questions)]
        train_df.drop(columns=['combined'], inplace=True)
        filtered_other_df.drop(columns=['combined'], inplace=True)
        group_sizes = filtered_other_df.groupby('question1')['is_duplicate'].transform('size')
        filtered_other_df = filtered_other_df[group_sizes >= 10]
        return filtered_other_df

    @staticmethod   
    def index_data(df, type_df="train"):
        # Indexer les données (dataframe) pour l'entrainement
        indexer = pt.DFIndexer("./index_" + type_df, overwrite=True)
        data_to_index = df.copy()
        data_to_index['docno'] = data_to_index['global_docno']
        index_ref = indexer.index(data_to_index['question1'], data_to_index['question2'], data_to_index['docno'])
        return index_ref

    @staticmethod
    def load_data(path,size=None):
        data = pd.read_csv(path)
        data = data.dropna() 
        # prendre un échantillon de taille size car il y'a 400k+ lignes
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
        # Tokenize les questions, taille = 128 sinon entrainement très lent
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer(question1, question2, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')