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
    def __init__(self, hidden_units):
        super(AttentionNetwork, self).__init__()
        self.hidden_units = hidden_units
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attn = nn.Linear(hidden_units * 3, hidden_units)
        self.v = nn.Linear(hidden_units, 1, bias=False)

    def forward(self, ht, encoder_out):
        # print("|== Attention Input Shape: ht, encoder_out", ht.shape, encoder_out.shape, ht.repeat(encoder_out.shape[1], 1 , 1).transpose(0, 1).shape, encoder_out.transpose(1, 2).shape)
        
        ht = ht.repeat(encoder_out.shape[1], 1 , 1)
        # print("ht.shape after repeat", ht.shape)
        ht = ht.transpose(0,1)
        # print("ht.shape after transpose", ht.shape)
        # encoder_out = encoder_out.transpose(1,2)
        # print("encoder_out.shape after transpose", encoder_out.shape)

        energy = torch.cat([ht, encoder_out], dim=2)
        # print("ENERGY SHAPE", energy.shape)
        energy = self.attn(energy)
        energy = self.relu(energy)        
        # print("ENERGY SHAPE POST TANH", energy.shape)
        attention = self.v(energy)
        # print("attention", attention.shape)
        attention = attention.squeeze(2)
        # print("attention shape post squeeze", attention.shape)
        attention_weights = self.softmax(attention)
        attention_weights = attention_weights.unsqueeze(1)
        # print("attention_weights.shape, encoder_out.shape", attention_weights.shape, encoder_out.shape)
        context = torch.bmm(attention_weights, encoder_out)
        # print("context shape", context.shape)
        return context, attention_weights

    

class LSTMAttnDecoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.5, bidirectional=False):
        super(LSTMAttnDecoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.attention = AttentionNetwork(hidden_units)
        self.LSTM = nn.LSTM(2*hidden_units + embed_dim, hidden_units, num_layers = num_layers, bidirectional = bidirectional, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_units, input_size)

    def forward(self, x, h0_c0, encoder_out):
        # print("|== Decoder Input Shape: x, h0_c0, encoder_out", x.shape, len(h0_c0), h0_c0[0].shape, h0_c0[1].shape, encoder_out.shape)
        x = self.dropout(self.embedding(x))
#         print("|== Decoder Embeddings Shape: x", x.shape)
        x = x.unsqueeze(1)

#         print("|== Decoder Embeddings unsqueezed(0) Shape: x", x.shape)
        context, attention_weights = self.attention(h0_c0[0], encoder_out)
        x = torch.cat([x, context], dim=2)
        decoder_out, (ht, ct) = self.LSTM(x, h0_c0)
#         print("|== Decoder Output Shape Shape: decoder_out, ht, ct", decoder_out.shape, ht.shape, ct.shape)
        
        out = self.fc(decoder_out)
#         print("|== Decoder FC OUT Shape: out", out.shape)
        
        return out, (ht, ct)




class AttentionNetworkBert(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionNetworkBert, self).__init__()
        self.hidden_units = hidden_units
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attn = nn.Linear(hidden_units * 2, hidden_units)
        self.v = nn.Linear(hidden_units, 1, bias=False)

    def forward(self, ht, encoder_out):
        # print("|== Attention Input Shape: ht, encoder_out", ht.shape, encoder_out.shape, ht.repeat(encoder_out.shape[1], 1 , 1).transpose(0, 1).shape, encoder_out.transpose(1, 2).shape)
        
        ht = ht.repeat(encoder_out.shape[1], 1 , 1)
        # print("ht.shape after repeat", ht.shape)
        ht = ht.transpose(0,1)
        # print("ht.shape after transpose", ht.shape)
        # encoder_out = encoder_out.transpose(1,2)
        # print("encoder_out.shape after transpose", encoder_out.shape)

        energy = torch.cat([ht, encoder_out], dim=2)
        # print("ENERGY SHAPE", energy.shape)
        energy = self.attn(energy)
        energy = self.relu(energy)        
        # print("ENERGY SHAPE POST TANH", energy.shape)
        attention = self.v(energy)
        # print("attention", attention.shape)
        attention = attention.squeeze(2)
        # print("attention shape post squeeze", attention.shape)
        attention_weights = self.softmax(attention)
        attention_weights = attention_weights.unsqueeze(1)
        # print("attention_weights.shape, encoder_out.shape", attention_weights.shape, encoder_out.shape)
        context = torch.bmm(attention_weights, encoder_out)
        # print("context shape", context.shape)
        return context, attention_weights

    

class LSTMAttnDecoderBert(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.5, bidirectional=False):
        super(LSTMAttnDecoderBert, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.attention = AttentionNetworkBert(hidden_units)
        self.LSTM = nn.LSTM(hidden_units + embed_dim, hidden_units, num_layers = num_layers, bidirectional = bidirectional, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_units, input_size)

    def forward(self, x, h0_c0, encoder_out):
        # print("|== Decoder Input Shape: x, h0_c0, encoder_out", x.shape, len(h0_c0), h0_c0[0].shape, h0_c0[1].shape, encoder_out.shape)
        x = self.dropout(self.embedding(x))
        # print("|== Decoder Embeddings Shape: x", x.shape)
        x = x.unsqueeze(1)

#         print("|== Decoder Embeddings unsqueezed(0) Shape: x", x.shape)
        context, attention_weights = self.attention(h0_c0[0], encoder_out)
        # print("|== Shape before concat x and context: ", x.shape, context.shape)
        x = torch.cat([x, context], dim=2)
        decoder_out, (ht, ct) = self.LSTM(x, h0_c0)
        # print("|== Decoder Output Shape Shape: decoder_out, ht, ct", decoder_out.shape, ht.shape, ct.shape)
        
        out = self.fc(decoder_out)
        # print("|== Decoder FC OUT Shape: out", out.shape)
        
        return out, (ht, ct)

