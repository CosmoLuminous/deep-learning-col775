import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.5):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers = num_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_units, input_size)
        
    def forward(self, x, h0_c0):
#         print("|== Decoder Input Shape: x, h0_c0", x.shape, len(h0_c0), h0_c0[0].shape, h0_c0[1].shape)
        x = self.dropout(self.embedding(x))
#         print("|== Decoder Embeddings Shape: x", x.shape)
        x = x.unsqueeze(1)
#         print("|== Decoder Embeddings unsqueezed(0) Shape: x", x.shape)
        decoder_out, (ht, ct) = self.LSTM(x, h0_c0)
#         print("|== Decoder Output Shape Shape: decoder_out, ht, ct", decoder_out.shape, ht.shape, ct.shape)
        
        out = self.fc(decoder_out)
#         print("|== Decoder FC OUT Shape: out", out.shape)
        
        return out, (ht, ct)

class AttentionNetwork(nn.Module):
        


class LSTMAttnDecoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.5):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers = num_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_units, input_size)
        
    def forward(self, x, h0_c0):
#         print("|== Decoder Input Shape: x, h0_c0", x.shape, len(h0_c0), h0_c0[0].shape, h0_c0[1].shape)
        x = self.dropout(self.embedding(x))
#         print("|== Decoder Embeddings Shape: x", x.shape)
        x = x.unsqueeze(1)
#         print("|== Decoder Embeddings unsqueezed(0) Shape: x", x.shape)
        decoder_out, (ht, ct) = self.LSTM(x, h0_c0)
#         print("|== Decoder Output Shape Shape: decoder_out, ht, ct", decoder_out.shape, ht.shape, ct.shape)
        
        out = self.fc(decoder_out)
#         print("|== Decoder FC OUT Shape: out", out.shape)
        
        return out, (ht, ct)