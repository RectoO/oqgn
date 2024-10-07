import torch


class LayoutLMv3AndFeaturesHeadClassification(torch.nn.Module):
    """
    Head for sentence-level classification tasks. Reference: ClassificationHead
    """

    def __init__(self, config, num_labels, n_tags):
        super().__init__()

        hidden_size = config.hidden_size
        classifier_dropout = config.hidden_dropout_prob

        self.dense = torch.nn.Linear(hidden_size + n_tags, hidden_size)
        self.dense_feat = torch.nn.Linear(n_tags, n_tags)
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x, y):
        self.eval()
        y = self.dense_feat(y)
        x = torch.cat([x, y], dim=2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
