import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>", "<num_value>", "<str_value>"]
SQL_KEYWORDS = ["t"+str(i+1) for i in range(10)] + [".", ",", "(", ")", "in", "not", "and", "between", "or", "where",
            "except", "union", "intersect",
            "group", "by", "order", "limit", "having","asc", "desc",
            "count", "sum", "avg", "max", "min",
           "<", ">", "=", "!=", ">=", "<=",
            "like",
            "distinct","*",
            "join", "on", "as", "select", "from"
           ] 
           
SQL_KEYWORDS = dict(zip(SQL_KEYWORDS, [10]*len(SQL_KEYWORDS)))
class GloveEmbeddings():
    def __init__(self, embed_dim, word2idx):
        self.embed_dim = embed_dim
        self.word2idx = word2idx
        self.special_tokens = SPECIAL_TOKENS
        self.vocab_size = len(word2idx)
    
    def get_embedding_matrix(self):
        # Load pre-trained GloVe embeddings
        glove = GloVe(name='6B', dim=self.embed_dim)
        embedding_matrix = torch.zeros((self.vocab_size, self.embed_dim))

        embedding_matrix[0] = torch.zeros(self.embed_dim)    # Padding token
        for i in range(1,len(SPECIAL_TOKENS)):            
            embedding_matrix[i] = torch.randn(self.embed_dim)    # Start-of-sentence token
            
        for k, v in self.word2idx.items():
            if k in SPECIAL_TOKENS:
                continue
            else:            
                if k in glove.stoi:
                    embedding_matrix[v] = torch.tensor(glove.vectors[glove.stoi[k]])
                else:
                    embedding_matrix[v] = embedding_matrix[1]
#                     print("unknown token", v)

        return embedding_matrix


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.5, bidirectional=False, embed_matrix=None):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.bidirectional = bidirectional
        self.embed_matrix = None
        if self.embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers = num_layers, dropout=p, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.hidden = nn.Linear(2*hidden_units, hidden_units)
            self.cell = nn.Linear(2*hidden_units, hidden_units)
    def forward(self, x):
#         print("ENCODER INPUT SHAPE", x.shape)
        x = self.dropout(self.embedding(x))
#         print("ENCODER EMBEDDING SHAPE", x.shape)
        
        encoder_out, (ht, ct) = self.LSTM(x)        
#         print("ENCODER OUTPUT SHAPE: encoder_out, ht, ct", encoder_out.shape, ht.shape, ct.shape)
        if self.bidirectional:
            # concatenate the forward and backward LSTM hidden states
            ht = self.hidden(torch.cat((ht[0:1], ht[1:2]), dim=2))
            ct = self.cell(torch.cat((ct[0:1], ct[1:2]), dim=2))
        return encoder_out, (ht, ct)


# class BiLSTMEncoder(nn.Module):
#     def __init__(self, input_size, embed_dim, hidden_units=1024, num_layers=1, p = 0.5, embed_matrix=None):
#         super(LSTMEncoder, self).__init__()
#         self.input_size = input_size
#         self.embed_dim = embed_dim
#         self.hidden_units = hidden_units
#         self.num_layers = num_layers
#         self.dropout = nn.Dropout(p)
#         self.bidirectional = True
#         self.embed_matrix = None
#         if self.embed_matrix is not None:
#             self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
#         else:
#             self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
#         self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers = num_layers, dropout=p, batch_first=True, bidirectional=True)
#         self.hidden = nn.Linear(2*hidden_units, hidden_units)
#         self.cell = nn.Linear(2*hidden_units, hidden_units)
#     def forward(self, x):
# #         print("ENCODER INPUT SHAPE", x.shape)
#         x = self.dropout(self.embedding(x))
# #         print("ENCODER EMBEDDING SHAPE", x.shape)
        
#         encoder_out, (ht, ct) = self.LSTM(x)        
# #         print("ENCODER OUTPUT SHAPE: encoder_out, ht, ct", encoder_out.shape, ht.shape, ct.shape)
#         if self.bidirectional:
#             # concatenate the forward and backward LSTM hidden states
#             ht = hidden(torch.cat((ht[0:1], ht[1:2])), dim=2)
#             ct = hidden(torch.cat((ct[0:1], ct[1:2])), dim=2)
#         return encoder_out, (ht, ct)