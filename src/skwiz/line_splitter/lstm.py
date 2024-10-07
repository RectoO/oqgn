import torch


class LSTMClassifier(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers=1, dropout=0.3, bidirectional=False
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        outputs = self.fc(lstm_out)

        return outputs
