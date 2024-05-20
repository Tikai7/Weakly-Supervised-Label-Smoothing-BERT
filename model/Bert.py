import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", output_size=2, dropout_rate=0.5) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        classifier_dropout = (
            self.bert.config.classifier_dropout if self.bert.config.classifier_dropout is not None else self.bert.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, inputs, attention_mask):
        outputs = self.bert(inputs, attention_mask)
        pooled_outputs = outputs.pooler_output
        logits = self.dropout(pooled_outputs)
        logits = self.fc(logits)
        return logits
