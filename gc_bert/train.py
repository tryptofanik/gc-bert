import argparse
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig

from gc_bert import log
from gc_bert.bert import BERT
from gc_bert.dataset import PubmedDataset
from gc_bert.gat.models import GAT, SpGAT
from gc_bert.gcn.models import GCN
from gc_bert.utils import accuracy
from gc_bert.trainer import GNNTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BERT_MODEL_NAME = 'bert-base-uncased'

logger = log.create_logger()


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
    loss = torch.tensor(losses).mean()
    logger.info(
        f'Epoch: {epoch+1:04d} loss_{mode}: {loss:.4f} acc_{mode}: {acc.item():.4f} '
        f'time: {(time.time() - t):.4f}s'
    )


def train_bert(model, dataset, epochs=1000):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    tokenizer_ = partial(
        tokenizer, padding='max_length', truncation=True, max_length=512, return_tensors='pt'
    )

    # manage dataset
    dataset.transform = tokenizer_
    n = len(dataset)
    n_train = int(n * 0.7)
    n_valid = int(n * 0.2)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        lengths=[n_train, n_valid, n - (n_train + n_valid)]
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)

    # run model
    for epoch in range(epochs):
        # train
        run_epoch(model, train_loader, optimizer, epoch, 'train')

        # valid
        if epoch % 10 == 0:
            run_epoch(model, valid_loader, optimizer, epoch, 'valid')      

    # test
    run_epoch(model, test_loader, optimizer, epoch, 'test')


def main(args):
    
    os.makedirs(args.save_dir, exist_ok=True)
    log.add_file_handler(logger, args.save_dir)
    logger.info(f'Parameters of run: {args}')

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
        trainer = GNNTrainer(model, dataset, logger, lr=0.001)
    elif args.model == 'gat':
        model = SpGAT(
            nfeat=512,
            nhid=256,
            nclass=3,
            dropout=0.5, 
            alpha=0.2,
            nheads=8
        )
        trainer = GNNTrainer(model, dataset, logger, lr=0.001)
    elif args.model == 'bert':
        config = BertConfig(num_labels=len(dataset.labels.unique()))
        if args.model_load_path is not None:
            model = BERT(config).from_pretrained(args.model_load_path, config=config)
        else:
            model = BERT(config).from_pretrained(BERT_MODEL_NAME, config=config)
        trainer = train_bert
    else:
        raise Exception('No model was chosen.')
    
    model = model.to(DEVICE)
    model = trainer.train(args.epochs)
    
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
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--weight-decay', default=0.01)
    args = parser.parse_args()
    
    main(args)
