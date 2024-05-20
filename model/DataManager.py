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
    def prepare_data(df, max_length=64, max_size=None):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if max_size:
            df = df.sample(n=max_size)
        # Extract columns as lists
        queries = df['query'].tolist()
        docs = df['docno'].tolist()  # Ensure this column corresponds to document texts
        labels = df['label'].tolist()
        # Concatenate queries and documents with the SEP token
        query_doc_pairs = [query + " [SEP] " + doc for query, doc in zip(queries, docs)]
        # Use batch encoding for the concatenated query-doc pairs
        encoded = tokenizer.batch_encode_plus(
            query_doc_pairs,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        inputs = encoded['input_ids']
        attention_masks = encoded['attention_mask']
        
        # Convert labels to tensor
        labels = torch.tensor(labels)
        
        return inputs, attention_masks, labels