import torch.nn as nn
import transformers
from transformers.modeling_bert import BertModel

class Bert(transformers.modeling_bert.BertPreTrainedModel):
    def __init__(self, hidden_size=128, output_size=2, dropout_rate=0.5) -> None:
        super().__init__()
        self.bert = BertModel()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        outputs = self.bert(inputs)
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs
