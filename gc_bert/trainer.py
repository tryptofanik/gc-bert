from functools import partial
import time

import torch
from torch.nn import BCEWithLogitsLoss, NLLLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gc_bert.utils import to_torch_sparse, vectorize_texts
from gc_bert.monitor import Monitor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL_NAME = "bert-base-uncased"


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        logger,
        lr=5e-5,
        weight_decay=0.01,
        loss_fun=BCEWithLogitsLoss(reduction="none"),
    ):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fun = loss_fun
        self.logger = logger

        self.train_monitor = Monitor(logger, dataset.num_labels, "train")
        self.valid_monitor = Monitor(logger, dataset.num_labels, "valid")
        self.test_monitor = Monitor(logger, dataset.num_labels, "test")
        self.optimizer = None
        self.lr_scheduler = None

    def prepare(self):
        if self.optimizer is None or self.lr_scheduler is None:
            self.create_optimizer()

    def create_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",  # 'min' for loss; 'max' for acc
            patience=40,
            factor=0.1,  # new_lr = lr * factor
        )

    def get_dataloader(self, mode="train"):
        self.dataset.change_mode(mode=mode)
        return DataLoader(
            self.dataset, batch_size=32, shuffle=True if mode == "train" else False,
        )

    def validate(self, mode="valid"):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        loader = self.get_dataloader(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            for X, y in loader:
                output = self.model(X)
                preds = torch.argmax(output, dim=-1)
                loss = self.loss_fun(preds, y)
                monitor.update(preds, y, loss)
        return monitor.emit(time=time.time() - t,)

    def train(self, epochs):
        self.prepare()

        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            self.dataset.change_mode("train")
            for X, y in self.get_dataloader(mode="train"):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss_fun(output, y)
                loss.mean().backward()
                self.optimizer.step()

                preds = torch.argmax(output, dim=-1)
                self.train_monitor.update(preds, y, loss)

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=time.time() - t,
            )
            valid_res = self.validate(mode="valid")
            self.lr_scheduler.step(valid_res["loss"])

        return self.model


class GNNTrainer(Trainer):
    def __init__(
        self, model, dataset, logger, loss_fun=NLLLoss(reduction="none"), **kwargs
    ):
        super().__init__(model, dataset, logger, **kwargs)
        self.loss_fun = loss_fun

    def prepare(self):
        super().prepare()
        if self.dataset.G is None:
            self.dataset.create_graph()
        if self.dataset.adj is None:
            self.dataset.create_adj_matrix(to_sparse=True)
        self.X = vectorize_texts(
            self.dataset.articles.abstract.tolist(), to_sparse=False
        ).to(DEVICE)
        self.adj = to_torch_sparse(self.dataset.adj).to(DEVICE)

    def validate(self, mode="valid", epoch=None):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            y = self.dataset.labels
            output = self.model(self.X, self.adj)
            output = output[self.dataset.mask]
            loss = self.loss_fun(output, y)
            preds = torch.argmax(output, dim=-1)
            monitor.update(preds, y, loss)
        return monitor.emit(epoch=epoch, time=round(time.time() - t, 3),)

    def train(self, epochs):

        self.prepare()

        for epoch in range(epochs):
            t = time.time()

            self.model.train()
            self.dataset.change_mode("train")
            y = self.dataset.labels
            self.optimizer.zero_grad()
            output = self.model(self.X, self.adj)
            output = output[self.dataset.mask]
            loss = self.loss_fun(output, y)
            loss.mean().backward()
            self.optimizer.step()

            preds = torch.argmax(output, dim=-1)
            self.train_monitor.update(preds, y, loss)

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=round(time.time() - t, 3),
            )
            valid_res = self.validate(mode="valid", epoch=epoch)
            self.lr_scheduler.step(valid_res["loss"])

        return self.model


class BERTTrainer(Trainer):
    def __init__(
        self,
        model,
        dataset,
        logger,
        lr=0.00005,
        weight_decay=0.01,
        loss_fun=BCEWithLogitsLoss(reduction="none"),
    ):
        super().__init__(model, dataset, logger, lr, weight_decay, loss_fun)

    def prepare(self):
        super().prepare()
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.tokenizer = partial(
            tokenizer,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        self.dataset.transform = self.tokenizer

    def validate(self, epoch=-1, mode="valid"):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        loader = self.get_dataloader(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            for X, y in loader:
                output = self.model(
                    input_ids=X["input_ids"].squeeze().to(DEVICE),
                    token_type_ids=X["token_type_ids"].squeeze().to(DEVICE),
                    attention_mask=X["attention_mask"].squeeze().to(DEVICE),
                    labels=y.to(DEVICE),
                )
                preds = torch.argmax(output.logits, dim=-1)
                monitor.update(preds, y, output.loss)
        return monitor.emit(time=time.time() - t, epoch=epoch)

    def train(self, epochs):
        self.prepare()

        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            self.dataset.change_mode("train")
            for X, y in self.get_dataloader(mode="train"):
                self.optimizer.zero_grad()
                output = self.model(
                    input_ids=X["input_ids"].squeeze().to(DEVICE),
                    token_type_ids=X["token_type_ids"].squeeze().to(DEVICE),
                    attention_mask=X["attention_mask"].squeeze().to(DEVICE),
                    labels=y.to(DEVICE),
                )
                output.loss.backward()
                preds = torch.argmax(output.logits, dim=-1)
                self.optimizer.step()
                self.train_monitor.update(preds, y, output.loss)

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=time.time() - t,
            )
            torch.cuda.empty_cache()

            valid_res = self.validate(epoch=epoch, mode="valid")
            self.lr_scheduler.step(valid_res["loss"])

        return self.model


class ComposedGraphBERTTrainer(Trainer):
    def __init__(
        self,
        model,
        dataset,
        logger,
        lr=0.00005,
        weight_decay=0.01,
        loss_fun=BCEWithLogitsLoss(reduction="none"),
    ):
        super().__init__(model, dataset, logger, lr, weight_decay, loss_fun)
        # text representations from BERT
        self.T = torch.empty((dataset.real_len, 768), dtype=torch.float32, device=DEVICE)
        # node representations from GNN
        self.N = torch.empty((dataset.real_len, 768), dtype=torch.float32, device=DEVICE)


    def fill_text_repr(self):
        with torch.no_grad():
            for mode in ['train', 'valid', 'test']:
                for X, _, idx in self.get_dataloader(mode=mode):
                    output = self.model.bert(
                        input_ids=X["input_ids"].squeeze().to(DEVICE),
                        token_type_ids=X["token_type_ids"].squeeze().to(DEVICE),
                        attention_mask=X["attention_mask"].squeeze().to(DEVICE),
                    )
                    self.T[idx, :] = output.last_hidden_state

    def fill_node_repr(self):
        with torch.no_grad():
            self.N = self.model.gnn(self.T, self.adj)

    def prepare(self):
        super().prepare()
        if self.dataset.G is None:
            self.dataset.create_graph()
        if self.dataset.adj is None:
            self.dataset.create_adj_matrix(to_sparse=True)
        self.X = vectorize_texts(
            self.dataset.articles.abstract.tolist(), to_sparse=False
        ).to(DEVICE)
        self.adj = to_torch_sparse(self.dataset.adj).to(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.tokenizer = partial(
            tokenizer,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        self.dataset.transform = self.tokenizer
        self.dataset.set_return_index(True)

        self.fill_text_repr()
        self.fill_node_repr()

    def validate(self, epoch=-1, mode="valid"):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        loader = self.get_dataloader(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            for E, y, idx in loader:
                logits = self.model(E, self.T, self.N, self.adj, idx)
                loss = self.loss_fun(logits)
                preds = torch.argmax(logits, dim=-1)
                monitor.update(preds, y, loss)
        return monitor.emit(time=time.time() - t, epoch=epoch)

    def train(self, epochs):
        self.prepare()

        for epoch in range(epoch):
            t = time.time()
            self.model.train()
            self.dataset.change_mode("train")
            for E, y, idx in self.get_dataloader(mode="train"):
                self.optimizer.zero_grad()
                logits = self.model(E, self.T, self.N, self.adj, idx)
                loss = self.loss_fun(logits)
                loss.backward()
                preds = torch.argmax(logits, dim=-1)
                self.optimizer.step()
                self.train_monitor.update(preds, y, loss)

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=time.time() - t,
            )
            torch.cuda.empty_cache()

            valid_res = self.validate(epoch=epoch, mode="valid")
            self.lr_scheduler.step(valid_res["loss"])

        return self.model