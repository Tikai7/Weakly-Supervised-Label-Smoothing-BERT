import torch.nn as nn
from transformers import BertModel
class BertForQuestionPairClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        # layer de dropout pour éviter overfitting, car le modèle overfit très rapidement
        self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits