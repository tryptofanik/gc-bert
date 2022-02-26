import json

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset

from gc_bert.utils import split


class PubmedDataset(Dataset):
    
    def __init__(self, data_path, citation_path, transform=None, target_transform=None, return_idx=False):
        self.articles = None
        self.citations = None
        self.G = None
        self.adj = None
        self.path = data_path
        self.citation_path = citation_path
        self.transform = transform
        self.target_transform = target_transform
        self.mode = None
        self.mask = None
        self.return_idx = return_idx

    def remove_disconnected_nodes(self):
        to_exclude = set(self.articles.index) - set(self.citations.target.astype(int)) - set(self.citations.source.astype(int))
        self.articles = self.articles.loc[~self.articles.index.isin(to_exclude)]
        self.pmid_to_id = {k:v for k,v in self.pmid_to_id.items() if v not in to_exclude}

    def clean_data(self):
        self.articles = (
            self.articles
            .loc[(self.articles.label.notna()) & (~self.articles.pmid.duplicated())
                & (self.articles.abstract.notna())]
            .pipe(lambda df: df.assign(
                label=df.label.astype(int),
                pmid=df.pmid.astype(int),
                time=pd.to_datetime(self.articles.history.apply(lambda x: min(x.values()))),
            ))
            .reset_index(drop=True)
            [['abstract', 'label', 'pmid', 'authors', 'title', 'time']]
        )
        self.pmid_to_id = (
            self.articles
            .reset_index()
            .set_index('pmid')
            ['index']
            .to_dict()
        )
        self.citations = (
            self.citations
            .dropna()
            .loc[(self.citations.source.isin(self.articles.pmid)) &
                (self.citations.target.isin(self.articles.pmid)) &
                (~self.citations.duplicated())]
            .assign(
                source=self.citations.source.map(self.pmid_to_id),
                target=self.citations.target.map(self.pmid_to_id),
            )
        )
        self.remove_disconnected_nodes()
    
    def load_data(self):
        with open(self.path) as f:
            self.articles = pd.DataFrame(json.load(f))
        self.citations = pd.read_csv(self.citation_path)
        self.clean_data()
        self.split_data()

    def split_data(self):
        idx_train, idx_valid, idx_test = split(self.articles.shape[0])
        self.articles.loc[idx_train, 'mode'] = 'train'
        self.articles.loc[idx_valid, 'mode'] = 'valid'
        self.articles.loc[idx_test, 'mode'] = 'test'

    def change_mode(self, mode):
        if mode in ['train', 'valid', 'test']:
            self.mode = mode
            self.mask = (self.articles['mode'] == mode).values
        else:
            raise Exception(f'Inproper mode {mode}')
    
    def set_return_idx(self, return_idx):
        self.return_idx = return_idx

    def create_authors_list(self):
        self.authors = set.union(*[set(i) for i in self.articles.authors.tolist()])

    def create_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.articles.index.tolist())
        self.G.add_edges_from(self.citations.values.tolist())
        return self.G

    def create_adj_matrix(self, to_sparse=True):
        if self.G is None:
            self.create_graph()

        if to_sparse:
            self.adj = nx.convert_matrix.to_scipy_sparse_matrix(
                self.G, nodelist=self.articles.index.tolist(), dtype='float32',
            )
        else:
            self.adj = nx.convert_matrix.to_numpy_array(
                self.G, nodelist=self.articles.index.tolist()
            )
        return self.adj 

    @property
    def labels(self):
        if self.mode is None:
            return torch.tensor(self.articles.label.tolist())
        else:
            return torch.tensor(
                self.articles.loc[self.mask].label.tolist()
            )

    @property
    def num_labels(self):
        return len(self.articles.label.unique())

    @property
    def real_len(self):
        return len(self.articles)

    def __len__(self):
        if self.mode is None:
            return len(self.articles)
        else:
            return len(self.articles.loc[self.mask])

    def __getitem__(self, idx):
        real_idx, text, label = self.articles.loc[
            self.mask, ['abstract', 'label']].reset_index().iloc[idx]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        if self.return_idx:
            return text, label, real_idx
        else:
            return text, label
