import torch


class LayoutLMv3ClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks. Reference: RobertaClassificationHead"""

    def __init__(self, config, num_labels):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        self.eval()
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
