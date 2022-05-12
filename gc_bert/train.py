import argparse
import os

import torch
from transformers import BertConfig

from gc_bert import log
from gc_bert.bert import BERT
from gc_bert.dataset import PubmedDataset
from gc_bert.gat import GAT
from gc_bert.gcn import GCN, GCN2
from gc_bert.gin import GIN
from gc_bert.bert_gnn import ComposedGraphBERT, ParallelGraphBERT, GCBERT
from gc_bert.trainer import BERTTrainer, GNNTrainer, ComposedGraphBERTTrainer

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
            dropout=0
        ).to(DEVICE)
        trainer = GNNTrainer(model, dataset, logger, lr=0.001)

    elif args.model == 'gat':
        model = GAT(
            nfeat=512,
            nhid=256,
            nclass=3,
            dropout=0.5, 
#             alpha=0.2,
            nheads=8
        ).to(DEVICE)
        trainer = GNNTrainer(model, dataset, logger, lr=0.001)
    
    elif args.model == 'gin':
        model = GIN(
            nfeat=512,
            nhid=512, 
            nout=3,
            dropout=0.5, 
        ).to(DEVICE)
        trainer = GNNTrainer(model, dataset, logger, lr=0.001)

    elif 'bert' in args.model:
        config = BertConfig(num_labels=len(dataset.labels.unique()))
        model_source = args.model_load_path if args.model_load_path is not None else BERT_MODEL_NAME

        if args.model == 'bert':
            model = BERT(config).from_pretrained(model_source, config=config).to(DEVICE)
            trainer = BERTTrainer(model, dataset, logger)

        elif args.model == 'bert-gcn':
            model = ComposedGraphBERT(config, model_source, dataset.real_len).to(DEVICE)
            trainer = ComposedGraphBERTTrainer(model, dataset, logger, lr=0.0001)

        elif args.model == 'bert-gcn-par':
            model = ParallelGraphBERT(config, model_source, dataset.real_len).to(DEVICE)
            trainer = ComposedGraphBERTTrainer(model, dataset, logger, lr=0.0001)
        
        elif args.model == 'gcbert':
            model = GCBERT(config, model_source, dataset.real_len).to(DEVICE)
            trainer = ComposedGraphBERTTrainer(model, dataset, logger, lr=0.00005)

    trainer.train(args.epochs)
    
    if args.model != 'bert':
        with open(os.path.join(args.save_dir, args.run_name + '.pt'), 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        model.save_pretrained(os.path.join(args.save_dir, args.run_name))
    dataset.articles.to_csv(os.path.join(args.save_dir, 'data.csv'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pubmed')
    parser.add_argument('--model')
    parser.add_argument('--model-load-path', default=None)
    parser.add_argument('--run-name', default='test')
    parser.add_argument('--save-dir', default='saved_models/')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', default=0.01)
    args = parser.parse_args()
    
    main(args)

