import sys
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report

class FinetuneWithCNNv1(pl.LightningModule):

    def __init__(self,
                 model,
                 learning_rate=2e-5,
                 bert_layers=4,
                 out_channels=128,
                 hidden_size=768,
                 kernel_sizes=[3, 4, 5],
                 ) -> None:

        super(FinetuneWithCNNv1, self).__init__()
        self.model = model
        self.lr = learning_rate
        self.bert_layers = bert_layers

        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1)) for kernel_size in kernel_sizes
        ])

        self.linear = nn.Linear(bert_layers * hidden_size, hidden_size)
        self.classifier = nn.Linear((out_channels * len(kernel_sizes)), 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        hs_output = torch.cat(model_output.hidden_states[-self.bert_layers:], dim=-1)
        hs_output = self.linear(hs_output)

        prepared_conv_input = hs_output.permute(0, 2, 1)

        out_conv = []

        for conv in self.conv1d:
            x = conv(prepared_conv_input)
            x = F.max_pool1d(self.relu(x), x.size(2))
            out_conv.append(x)

        logits = torch.cat(out_conv, 1).squeeze(dim=-1)
        classifier_out = self.classifier(self.dropout(logits))

        return classifier_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        loss = self.criterion(outputs, targets)

        metrics = {}
        metrics['train_loss'] = loss.item()

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in validation_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['val_loss'] = loss.item()
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print()
        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def test_epoch_end(self, test_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in test_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['test_loss'] = loss.item()
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        loss = self.criterion(outputs, targets)

        true = targets.to(torch.device("cpu"))
        pred = (self.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        pred = (self.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return pred[0]

class FinetuneWithCNNv2(pl.LightningModule):

    def __init__(self,
                 model,
                 learning_rate=2e-5,
                 bert_layers=4,
                 out_channels=128,
                 hidden_size=768,
                 kernel_sizes=[3, 4, 5],
                 ) -> None:

        super(FinetuneWithCNNv2, self).__init__()
        self.model = model
        self.lr = learning_rate
        self.bert_layers = bert_layers

        self.conv2d = nn.ModuleList([
            nn.Conv2d(in_channels=bert_layers, out_channels=out_channels, kernel_size=[kernel_size, hidden_size], padding=(kernel_size - 1, 0)) for kernel_size in kernel_sizes
        ])

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear((out_channels * len(kernel_sizes)), 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        hs_output = torch.stack(model_output.hidden_states[-self.bert_layers:], dim=1)

        out_conv = []

        for conv in self.conv2d:
            x = conv(hs_output)
            x = self.relu(x)
            x = x.squeeze(-1)
            x = F.max_pool1d(x, x.size(2))
            out_conv.append(x)

        logits = torch.cat(out_conv, 1).squeeze(dim=-1)
        classifier_out = self.classifier(self.dropout(logits))

        return classifier_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        loss = self.criterion(outputs, targets)

        metrics = {}
        metrics['train_loss'] = loss.item()

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in validation_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['val_loss'] = loss.item()
        metrics['val_accuracy'] = accuracy
        metrics['val_f1_score'] = f1_score
        metrics['val_precision'] = precision
        metrics['val_recall'] = recall

        print()
        print(metrics)

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, true, pred = self._shared_eval_step(batch, batch_idx)
        return loss, true, pred
    
    def test_epoch_end(self, test_step_outputs):
        loss = torch.Tensor().to(device='cuda')
        true = []
        pred = []

        for output in test_step_outputs:
            loss = torch.cat((loss, output[0].view(1)), dim=0)
            true += output[1].numpy().tolist()
            pred += output[2].numpy().tolist()

        loss = torch.mean(loss)

        cls_report = classification_report(true, pred, labels=[0, 1], output_dict=True, zero_division=0)

        accuracy = cls_report['accuracy']
        f1_score = cls_report['1']['f1-score']
        precision = cls_report['1']['precision']
        recall = cls_report['1']['recall']

        metrics = {}
        metrics['test_loss'] = loss.item()
        metrics['test_accuracy'] = accuracy
        metrics['test_f1_score'] = f1_score
        metrics['test_precision'] = precision
        metrics['test_recall'] = recall

        self.log_dict(metrics, prog_bar=False, on_epoch=True)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        loss = self.criterion(outputs, targets)

        true = targets.to(torch.device("cpu"))
        pred = (self.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return loss, true, pred

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        outputs = torch.squeeze(self(input_ids=input_ids, attention_mask=attention_mask), dim=1)

        pred = (self.sigmoid(outputs) >= 0.5).int().to(torch.device("cpu"))

        return pred[0]