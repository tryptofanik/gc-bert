import argparse
import time
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


def train_gcn(model, dataset, epochs=1000):
    
    labels = dataset.labels
    adj = dataset.create_adj_matrix()
    X = vectorize_texts(dataset.articles.abstract.fillna('').tolist())
    
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.01)
    
    idx_train, idx_valid, idx_test = split(len(dataset))
    
    for epoch in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(X, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_valid], labels[idx_valid])
        acc_val = accuracy(output[idx_valid], labels[idx_valid])

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        
def main(args):
    
    if args.dataset == 'pubmed':
        dataset = PubmedDataset('pubmed/data/articles.json', 'pubmed/data/citations.csv')
        dataset.load_data()
    
    if args.model == 'gcn':
        import pygcn 
        model = pygcn.GCN(nfeat=512,
            nhid=512,
            nclass=3,
            dropout=0.5)
    
    train_gcn(model, dataset)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--weight_decay', default=0.01)
    args = parser.parse_args()
    
    main(args)
    