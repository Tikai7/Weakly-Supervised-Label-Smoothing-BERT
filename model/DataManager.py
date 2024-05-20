import torch
from transformers import BertTokenizer

class DataManager():
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }
    
    @staticmethod
    def prepare_data(df, max_length=512):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = []
        attention_masks = []
        labels = []
        for _, row in df.iterrows():
            encoded = tokenizer.encode_plus(
                row['title','docno'], 
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            inputs.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(row['label'])
        return torch.cat(inputs, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)
