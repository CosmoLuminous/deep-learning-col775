import argparse
import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import json 
from time import time
from tqdm import tqdm
from datetime import datetime
from torchtext.vocab import GloVe

from utils import *
from dataset import *
from encoder import LSTMEncoder
from decoder import LSTMDecoder

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

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.model_type = args.model_type
        self.embed_dim = args.embed_dim        
        self.encoder_hidden_units = args.en_hidden
        self.decoder_hidden_units = args.de_hidden
        self.encoder_num_layers = args.en_num_layers
        self.decoder_num_layers = args.de_num_layers
        self.processed_data = args.processed_data
        self.encoder_word2idx = self.get_encoder_word2idx()
        self.decoder_word2idx = self.get_decoder_word2idx()
        self.encoder_input_size = len(self.encoder_word2idx)
        self.decoder_input_size = len(self.decoder_word2idx)
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        
        

    def get_encoder_word2idx(self):
        with open(os.path.join(self.processed_data, "encoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        
        return word2idx
    
    def get_decoder_word2idx(self):
        
        with open(os.path.join(self.processed_data, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)        
        
        return word2idx
    
    def get_encoder(self):
        print("Loading GloVe embeddings...")
        glove = GloveEmbeddings(self.embed_dim, self.encoder_word2idx)
        embedding_matrix = glove.get_embedding_matrix()
        print("Loading Encoder...")
        encoder = LSTMEncoder(input_size = self.encoder_input_size, embed_dim = self.embed_dim, 
                              hidden_units=self.encoder_hidden_units, num_layers=self.encoder_num_layers, p = 0.3, bidirectional=False, embed_matrix=embedding_matrix)
        
        return encoder
    
    def get_decoder(self):
        
        if self.model_type == "Seq2Seq":
            print("Loading Seq2Seq LSTM Decoder...")
            decoder = LSTMDecoder(input_size = self.decoder_input_size, embed_dim = self.embed_dim, 
                              hidden_units=self.decoder_hidden_units, num_layers=self.decoder_num_layers, p = 0.3)
        
        elif self.model_type == "Seq2SeqAttn":
            pass
        else:
            pass
        return decoder
        
    def forward(self, question, query, tf_ratio=0.5):
        batch_size = question.shape[0]
        target_len = query.shape[1]
        
        _, (hidden, cell) = self.encoder(question)
        
        target_vocab_size = self.decoder_input_size
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.args.device)
        words = torch.zeros(batch_size, target_len).to(self.args.device)
        x = query[:,0]
        words[:, 0] = query[:,0]
        for t in range(1, target_len):
            output, (hidden, cell) = self.decoder(x, (hidden, cell))
#             print("Seq2seq out shape", output.shape)
            output = output.squeeze(1)
            outputs[:,t,:] = output
            x = output.argmax(dim=1)
            words[:, t] = x
        
        return outputs, words