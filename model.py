import torch.nn as nn
import torch

class TransformerDetector(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.embedding = nn.Linear(128, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
