import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from gc_bert.bert import BERT
from gc_bert.gcn import GCN2
from gc_bert.gin import GIN
from gc_bert.gat import GAT
from gc_bert.sage import SAGE


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ComposedGraphBERT(nn.Module):

    def __init__(self, bert_config, pretrained, n):
        super().__init__()
        self.bert = BERT(bert_config).from_pretrained(pretrained)
        self.gnn = GCN2(
            nfeat=768,
            nhid=768,
            nclass=768,
            dropout=0.5
        )
        self.N = nn.parameter.Parameter(torch.FloatTensor(n, 768), requires_grad=False)
        self.T = nn.parameter.Parameter(torch.FloatTensor(n, 768), requires_grad=False)
        self.classifier = nn.Linear(768, bert_config.num_labels)

    def forward(self, E, edge_idx, idx=None):
        output = self.bert(
            input_ids=E["input_ids"].squeeze().to(DEVICE),
            token_type_ids=E["token_type_ids"].squeeze().to(DEVICE),
            attention_mask=E["attention_mask"].squeeze().to(DEVICE),
        )
        self.T[idx] = torch.clone(output['last_hidden_state'])
        N_ = self.gnn(self.T, edge_idx)
        if idx is not None:
            self.N[idx] = N_[idx]
        else:
            self.N = nn.parameter.Parameter(N_, requires_grad=False)
#         self.N = N_  # would it be better?
        logits = self.classifier(N_[idx])
        return logits


class ParallelGraphBERT(nn.Module):
    
    def __init__(self, bert_config, pretrained, n):
        super().__init__()
        self.bert = BERT(bert_config).from_pretrained(pretrained)
        self.gnn = GCN2(
            nfeat=768,
            nhid=768,
            nclass=768,
            dropout=0.5
        )
        self.N = nn.parameter.Parameter(torch.FloatTensor(n, 768), requires_grad=False)
        self.T = nn.parameter.Parameter(torch.FloatTensor(n, 768), requires_grad=False)
        self.classifier = nn.Linear(768 * 2, bert_config.num_labels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, E, edge_idx, idx):
        output = self.bert(
            input_ids=E["input_ids"].squeeze().to(DEVICE),
            token_type_ids=E["token_type_ids"].squeeze().to(DEVICE),
            attention_mask=E["attention_mask"].squeeze().to(DEVICE),
        )
        self.T[idx] = torch.clone(output['last_hidden_state'])
        N_ = self.gnn(self.T, edge_idx)
        self.N[idx] = N_[idx]
        merged = torch.cat([N_[idx], self.T[idx]], dim=1)
        logits = self.classifier(merged)
        return logits


class GCBERT(nn.Module):

    def __init__(self, bert_config, pretrained, n):
        super().__init__()
        self.bert = BERT(bert_config).from_pretrained(pretrained)
        self.gnn = SAGE(
            nfeat=768,
            nhid=768,
            nclass=768,
            dropout=0.5
        )
        self.N = nn.parameter.Parameter(torch.FloatTensor(n, 768), requires_grad=False)
        self.T = nn.parameter.Parameter(torch.FloatTensor(n, 768), requires_grad=False)
        self.classifier = nn.Linear(768, bert_config.num_labels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, E, edge_idx, idx=None):
        output = self.bert(
            input_ids=E["input_ids"].squeeze().to(DEVICE),
            # graph_embeddings=self.N[idx].unsqueeze(1).expand(idx.shape[0], 512, 768),
            graph_embeddings=self.N[idx],
            token_type_ids=E["token_type_ids"].squeeze().to(DEVICE),
            attention_mask=E["attention_mask"].squeeze().to(DEVICE),
        )
        self.T[idx] = torch.clone(output['last_hidden_state'])
        N_ = self.gnn(self.T, edge_idx)
        if idx is not None:
            self.N[idx] = N_[idx]
        else:
            self.N = nn.parameter.Parameter(N_, requires_grad=False)
        # merge = torch.cat([N_[idx], self.T[idx]], 1)
        merge = N_[idx] + self.T[idx]
        logits = self.classifier(self.dropout(merge))
        return logits
