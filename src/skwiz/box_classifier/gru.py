import torch
from typing import Dict


class GRUBoxClassifier(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 extraction_label2id: Dict[str, Dict[str, int]],
                 n_tags: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool
                 ):
        super().__init__()
        self.extraction_label2id = extraction_label2id
        self.classifiers = torch.nn.ModuleDict()
        self.input_size = input_size

        for key, labels in extraction_label2id.items():
            num_labels = len(labels)
            self.classifiers[key] = GRUPredictionHead(
                input_size=input_size,
                n_tags=n_tags,
                num_labels=num_labels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )

    def forward(self, layoutlm_outputs: torch.Tensor, tags: torch.Tensor):
        logits = {}
        for key, classifier in self.classifiers.items():
            logits[key] = classifier(layoutlm_outputs, tags)
        return logits


class GRUPredictionHead(torch.nn.Module):
    def __init__(self, input_size: int, n_tags: int, num_labels: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.dense_feat = torch.nn.Linear(n_tags, n_tags)
        self.gru = torch.nn.GRU(input_size=input_size+n_tags, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(
            hidden_size*(2 if bidirectional else 1), num_labels)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y = self.dense_feat(y)
        x = torch.cat([x, y], dim=2)

        gru_out, _ = self.gru(x)
        outputs = self.fc(gru_out)

        return outputs
