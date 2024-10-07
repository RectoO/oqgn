import torch


class GRUClassifier(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers=1, dropout=0.3, bidirectional=False
    ):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        outputs = self.fc(gru_out)

        return outputs
