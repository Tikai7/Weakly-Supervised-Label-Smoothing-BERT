import torch.nn as nn
from transformers import BertModel
class BertForQuestionPairClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        classifier_dropout = (
            self.bert.config.classifier_dropout if self.bert.config.classifier_dropout is not None else self.bert.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits