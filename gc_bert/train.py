import argparse
import os

import torch
from transformers import BertConfig

from gc_bert import log
from gc_bert.bert import BERT
from gc_bert.dataset import PubmedDataset
from gc_bert.gat.models import SpGAT
from gc_bert.gcn.models import GCN
from gc_bert.trainer import BERTTrainer, GNNTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BERT_MODEL_NAME = 'bert-base-uncased'

logger = log.create_logger()


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
        trainer = BERTTrainer(model, dataset, logger)
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
    parser.add_argument('--dataset', default='pubmed')
    parser.add_argument('--model')
    parser.add_argument('--model-load-path', default=None)
    parser.add_argument('--run-name', default='test')
    parser.add_argument('--save-dir', default='saved_models/')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--weight-decay', default=0.01)
    args = parser.parse_args()
    
    main(args)
