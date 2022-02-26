import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from gc_bert.bert import BERT
from gc_bert.gcn.models import GCN2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ComposedGraphBERT(nn.Module):

    def __init__(self, bert_config, pretrained):
        super().__init__()
        self.bert = BERT(bert_config).from_pretrained(pretrained)
        self.gnn = GCN2(
            nfeat=768,
            nhid=768,
            nout=768,
            dropout=0.5
        )
        self.classifier = nn.Linear(768, bert_config.num_classes)
        self.adj = None

    def forward(self, E, T, N, adj, idx):
        output = self.bert(
            input_ids=E["input_ids"].squeeze().to(DEVICE),
            token_type_ids=E["token_type_ids"].squeeze().to(DEVICE),
            attention_mask=E["attention_mask"].squeeze().to(DEVICE),
        )
        T[idx] = output.last_hidden_state
        N_ = self.gnn(T, adj)
        N[idx] = N_
        logits = self.classifier(N_)
        return logits
