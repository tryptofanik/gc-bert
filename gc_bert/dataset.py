import json

import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from gc_bert.utils import to_torch_sparse


class PubmedDataset(Dataset):
    
    def __init__(self, data_path, citation_path, transform=None, target_transform=None):
        self.labels = None
        self.articles = None
        self.texts = None
        self.citations = None
        self.G = None
        self.adj = None
        self.path = data_path
        self.citation_path = citation_path
        self.transform = transform
        self.target_transform = target_transform
        
    def clean_data(self):
        self.articles = (
            self.articles
            .loc[(self.articles.label.notna()) & (~self.articles.pmid.duplicated())]
            .pipe(lambda df: df.assign(
                label=df.label.astype(int),
                pmid=df.pmid.astype(int),
            ))
            .reset_index(drop=True)
            [['abstract', 'label', 'pmid', 'authors', 'title']]
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
    
    def load_data(self):
        with open(self.path) as f:
            self.articles = pd.DataFrame(json.load(f))
        self.citations = pd.read_csv(self.citation_path)
        self.clean_data()
        self.texts = self.articles.abstract.tolist()
        self.labels = torch.tensor(self.articles.label.tolist())
        
    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
#         text, label = self.articles.loc[idx, ['abstract', 'label']]
        text = self.texts[idx]
        label = self.labels[idx]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        return text, label     

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
                self.G, nodelist=self.articles.index.tolist()
            )
            self.adj = to_torch_sparse(self.adj)
        else:
            self.adj = nx.convert_matrix.to_numpy_array(
                self.G, nodelist=self.articles.index.tolist()
            )
            self.adj = torch.tensor(self.adj)
        return self.adj

