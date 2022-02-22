import argparse
import time
import os
import numpy as np

from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gc_bert.dataset import PubmedDataset
from gc_bert.utils import to_torch_sparse
from gat.models import SpGAT, GAT
from pygcn import GCN


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BERT_MODEL_NAME = 'bert-base-uncased'


def split(n):
    train_idx, test_idx = train_test_split(np.arange(n), train_size=0.7)
    valid_idx, test_idx = train_test_split(test_idx, train_size=1/3)

    idx_train = np.zeros(n).astype(bool)
    idx_train[train_idx] = True

    idx_valid = np.zeros(n).astype(bool)
    idx_valid[valid_idx] = True

    idx_test = np.zeros(n).astype(bool)
    idx_test[test_idx] = True
    return idx_train, idx_valid, idx_test
    

def vectorize_texts(texts, dim=512, min_df=0.001, max_df=0.5, to_sparse=True):
    vectorizer = TfidfVectorizer(
        strip_accents='ascii', max_df=max_df, min_df=min_df, max_features=dim
    )
    X = vectorizer.fit_transform(
        texts
    )
    if to_sparse:
        X = to_torch_sparse(X)
    else:
        X = torch.tensor(X.todense(), dtype=torch.double)
    return X


def accuracy(preds, labels):
    if len(preds.shape) == 2:
        preds = preds.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train_gn(model, X, adj, labels, epochs=1000):

    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.01)

    idx_train, idx_valid, idx_test = split(len(labels))
    
    for epoch in range(epochs):
        t = time.time()

        # train
        model.train()
        optimizer.zero_grad()
        output = model(X, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # validate
        model.eval()
        output = model(X, adj)
        loss_val = F.nll_loss(output[idx_valid], labels[idx_valid])
        acc_val = accuracy(output[idx_valid], labels[idx_valid])

        print(f'Epoch: {epoch+1:04d} '
              f'loss_train: {loss_train.item():.4f} '
              f'acc_train: {acc_train.item():.4f} '
              f'loss_val: {loss_val.item():.4f} '
              f'acc_val: {acc_val.item():.4f} '
              f'time: {(time.time() - t):.4f}s')
    
    model.eval()
    output = model(X, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    print(f"Test: loss_test: {loss_test:.4f} acc_test: {acc_test:.4f}")

    return model


def train_gat(model, dataset, epochs=1000):
    
    labels = dataset.labels.to(DEVICE)
    adj = dataset.create_adj_matrix(to_sparse=True)
    X = vectorize_texts(dataset.articles.abstract.fillna('').tolist(), to_sparse=False).to(DEVICE)
    model = train_gn(model, X, adj, labels, epochs)

    return model


def train_gcn(model, dataset, epochs=1000):
    
    labels = dataset.labels.to(DEVICE)
    adj = dataset.create_adj_matrix()
    X = vectorize_texts(dataset.articles.abstract.fillna('').tolist()).to(DEVICE)
    model = train_gn(model, X, adj, labels, epochs)

    return model


def run_epoch(model, loader, optimizer, epoch, mode='train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    labels = []
    predictions = []
    losses = []
    
    t = time.time()
    for X, y in loader:
        output = model(
            input_ids=X['input_ids'].squeeze().to(DEVICE),
            token_type_ids=X['token_type_ids'].squeeze().to(DEVICE),
            attention_mask=X['attention_mask'].squeeze().to(DEVICE),
            labels=y.to(DEVICE)
        )
        output.loss.backward()
        preds = torch.argmax(output.logits, dim=-1)
        labels.append(y)
        predictions.append(preds)
        losses.append(output.loss.item())
        optimizer.step()
        optimizer.zero_grad()
    
    acc = accuracy(torch.concat(predictions).to('cpu'), torch.concat(labels))
    loss = torch.concat(losses).mean()
    print(
        f'Epoch: {epoch+1:04d} loss_{mode}: {loss:.4f} acc_{mode}: {acc.item():.4f}'
        f'time: {(time.time() - t):.4f}s'
    )


def train_bert(model, dataset, epochs=1000):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    tokenizer_ = partial(
        tokenizer, padding='max_length', truncation=True, max_length=512, return_tensors='pt'
    )
    dataset.transform = tokenizer_
    idx_train, idx_valid, idx_test = split(len(dataset))

    train_loader = DataLoader(dataset[idx_train], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[idx_valid], batch_size=32)
    test_loader = DataLoader(dataset[idx_test], batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)

    for epoch in range(epochs):
        # train
        run_epoch(model, train_loader, optimizer, epoch, 'train')

        # valid
        if epoch % 10 == 0:
            run_epoch(model, valid_loader, optimizer, epoch, 'valid')      

    # test
    run_epoch(model, test_loader, optimizer, epoch, 'test')



def main(args):
    
    if args.dataset == 'pubmed':
        dataset = PubmedDataset('pubmed/data/articles.json', 'pubmed/data/citations.csv')
        dataset.load_data()
    else:
        raise Exception('No dataset was chosen.')
    
    if args.model == 'gcn':
        model = GCN(
            nfeat=512,
            nhid=256,
            nclass=3,
            dropout=0.5
        )
        train = train_gcn
    elif args.model == 'gat':
        model = SpGAT(
            nfeat=512,
            nhid=256,
            nclass=3,
            dropout=0.5, 
            alpha=0.2,
            nheads=8
        )
        train = train_gat
    elif args.model == 'bert':
        if args.model_load_path is not None:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_load_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL_NAME, 
                num_labels=len(dataset.labels.unique())
            )
        train = train_bert
    else:
        raise Exception('No model was chosen.')
    
    model = model.to(DEVICE)
    model = train(model, dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.model != 'bert':
        with open(os.path.join(args.save_dir, args.run_name + '.pt'), 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        model.save_pretrained(os.path.join(args.save_dir, args.run_name))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--model-load-path', default=None)
    parser.add_argument('--run-name', default='test')
    parser.add_argument('--save-dir', default='saved_models/')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--weight-decay', default=0.01)
    args = parser.parse_args()
    
    main(args)
