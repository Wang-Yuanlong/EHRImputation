import torch
from torch import nn

class ValueEmbedding(nn.Module):
    def __init__(self, var_num, input_dim=1, output_dim=64):
        super(ValueEmbedding, self).__init__()
        self.var_num = var_num
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embedding = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, var_num)
        b, l, v = x.size()
        x = x.unsqueeze(-1)
        x = self.embedding(x) 
        return x

class Imputer(nn.Module):
    def __init__(self, varible_num=27, hidden_dim=64, output_dim=1, n_layers=2, dropout=0.1, bidirectional=False):
        super(Imputer, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.D = 2 if bidirectional else 1

        self.embedding = ValueEmbedding(varible_num, output_dim=hidden_dim)
        self.mapping = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, hidden_dim)
                        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.decoder = nn.Sequential(
                            nn.Linear(self.D * hidden_dim, hidden_dim // 2),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            # nn.Linear(hidden_dim // 2, output_dim),
                            nn.Linear(hidden_dim // 2, varible_num)
                        )

    def forward(self, x, mask):
        # x: (batch_size, seq_len, varible_num)
        x = self.embedding(x)
        x = self.mapping(x)
        x = x * mask.unsqueeze(-1)

        # x: (batch_size, seq_len, varible_num, hidden_dim)
        b, l, v, h = x.size()
        x = x.view(b * l, v, h)
        x = self.pooling(x.transpose(-1, -2)).squeeze(-1)
        x = x.view(b, l, -1)

        # x: (batch_size, seq_len, hidden_dim)
        x, _ = self.lstm(x)
        x = self.decoder(x)
        
        # x: (batch_size, seq_len, varible_num)
        x = x * mask
        return x