import torch.nn as nn
from transformers import BertModel

class Bert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", output_size=2, dropout_rate=0.5) -> None:
        super(self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, inputs, attention_mask):
        outputs = self.bert(inputs, attention_mask)
        pooled_outputs = outputs.pooler_output
        logits = self.dropout(pooled_outputs)
        logits = self.fc(logits)
        return logits
