import argparse
import time
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from gc_bert.dataset import PubmedDataset
from gc_bert.utils import to_torch_sparse


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
    

def vectorize_texts(texts, dim=512, min_df=0.001, max_df=0.5):
    vectorizer = TfidfVectorizer(
        strip_accents='ascii', max_df=max_df, min_df=min_df, max_features=dim
    )
    X = vectorizer.fit_transform(
        texts
    )
    X = to_torch_sparse(X)
    return X


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train_gcn(model, dataset, epochs=100):
    
    labels = dataset.labels
    adj = dataset.create_adj_matrix()
    X = vectorize_texts(dataset.articles.abstract.fillna('').tolist())
    
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.01)
    
    idx_train, idx_valid, idx_test = split(len(dataset))
    
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


def main(args):
    
    if args.dataset == 'pubmed':
        dataset = PubmedDataset('pubmed/data/articles.json', 'pubmed/data/citations.csv')
        dataset.load_data()
    
    if args.model == 'gcn':
        import pygcn 
        model = pygcn.GCN(nfeat=512,
            nhid=256,
            nclass=3,
            dropout=0.5)
    
    model = train_gcn(model, dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(os.path.join(args.save_dir, args.run_name + '.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--run-name', default='test')
    parser.add_argument('--save-dir', default='saved_models/')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--weight-decay', default=0.01)
    args = parser.parse_args()
    
    main(args)
