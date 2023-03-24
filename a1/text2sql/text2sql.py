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
        self.start_token = self.decoder_word2idx["<sos>"]
        self.end_token = self.decoder_word2idx["<eos>"]
        
        

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
        max_target_len = query.shape[1]
        
        _, (hidden, cell) = self.encoder(question)

        target_vocab_size = self.decoder_input_size
        outputs = torch.zeros(batch_size, max_target_len, target_vocab_size).to(self.args.device)
        words = torch.zeros(batch_size, max_target_len).to(self.args.device)
        
        if self.args.search_type == "greedy":
            x = query[:,0]            
            words[:, 0] = query[:,0]
            for t in range(1, max_target_len):
                # print("DECODER INPUT SHAPES", x.shape, hidden.shape, cell.shape)
                output, (hidden, cell) = self.decoder(x, (hidden, cell))
    #             print("Seq2seq out shape", output.shape)
                output = output.squeeze(1)
                outputs[:,t,:] = output
                x = output.argmax(dim=1)
                words[:, t] = x
            return outputs, words
        
        elif self.args.search_type == "beam":
            # print("Encoder hidden and cell shape", hidden.shape, cell.shape)
            for b in range(batch_size):
                outputs[b,:,:], words[b,:] = beam_search(self.args, self.decoder, hidden[:,b,:].unsqueeze(1), cell[:,b,:].unsqueeze(1), self.start_token, self.end_token, target_vocab_size, max_target_len, 3)
            return outputs, words
        else:
            pass




def beam_search(args, decoder, en_ht, en_ct, start_token, end_token, target_vocab_size, max_target_len = 80, beam_size = 3):
    beam = [([start_token], (en_ht, en_ct), 0, [])]
    print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)
    top_sentences = []
    i = 0
    while i < max_target_len:
        new_beam = []
        for sequence, (ht, ct), score, outputs in beam:
            prev_token = [sequence[-1]] #get first token for each beam
            prev_token = torch.LongTensor(prev_token).to(args.device)
            # print("DECODER INPUT SHAPES", prev_token, prev_token.shape, ht.shape, ct.shape)
            decoder_out, (ht, ct) = decoder(prev_token, (ht, ct)) #pass through decoder
            # print("DECODER OP SHAPES", decoder_out.shape, ht.shape, ct.shape)
            decoder_out = decoder_out.squeeze(1)

            outputs.append(decoder_out)
            
            top_info = decoder_out.topk(beam_size, dim=1) #get top k=beam_size possible word indices and their values
            top_vals, top_inds = top_info
            # print("TOP K VALS AND INDS", top_vals, top_inds)
            
            #create new candidates and append them in the new_beam.
            #Then select top k beams based on their corresponding scores
            for j in range(beam_size): 
                new_word_idx = top_inds[0][j]                
                new_seq = sequence + [new_word_idx.item()]
                new_word_prob = torch.log(top_vals[0][j])
                updated_score = score - new_word_prob                
                new_candidate = (new_seq, (ht, ct), updated_score, outputs)
                new_beam.append(new_candidate)
                # print(new_word_idx, print(updated_score))
            # print("len(new_beam)", len(new_beam))
           
        #select top k beams based on their corresponding scores
        new_beam = sorted(new_beam, reverse=False, key=lambda x: x[2])
        beam = new_beam[-beam_size:]
        # print("beam len", len(beam), beam[0][0], beam[0][2])
        i += 1 
    best_candidate = beam[0][0] #return best candidate based on score
    outputs = beam[0][3]
    decoder_outputs = torch.zeros(1, max_target_len, target_vocab_size).to(args.device)
    decoded_words = torch.zeros(1, max_target_len)
    print("max_target_len, len(outputs), len(best_candidate)", max_target_len, len(outputs), len(best_candidate))
    for t in range(1, max_target_len):
        decoder_outputs[:,t,:] = outputs[t]
        decoded_words[:,t] = torch.LongTensor([best_candidate[t]]).to(args.device)
    return decoder_outputs, best_candidate



def working_beam_search(args, decoder, en_ht, en_ct, start_token, end_token, max_target_len = 80, beam_size = 3):
    beam = [([start_token], (en_ht, en_ct), 0)]
    print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)
    top_sentences = []
    i = 0
    while i < max_target_len:
        new_beam = []
        for sequence, (ht, ct), score in beam:
            prev_token = [sequence[-1]] #get first token for each beam
            prev_token = torch.LongTensor(prev_token).to(args.device)
            print("DECODER INPUT SHAPES", prev_token, prev_token.shape, ht.shape, ct.shape)
            decoder_out, (ht, ct) = decoder(prev_token, (ht, ct)) #pass through decoder
            print("DECODER OP SHAPES", decoder_out.shape, ht.shape, ct.shape)
            decoder_out = decoder_out.squeeze(1)

            
            top_info = decoder_out.topk(beam_size, dim=1) #get top k=beam_size possible word indices and their values
            top_vals, top_inds = top_info
            print("TOP K VALS AND INDS", top_vals, top_inds)
            
            #create new candidates and append them in the new_beam.
            #Then select top k beams based on their corresponding scores
            for j in range(beam_size): 
                new_word_idx = top_inds[0][j]                
                new_seq = sequence + [new_word_idx.item()]
                new_word_prob = torch.log(top_vals[0][j])
                updated_score = score - new_word_prob
                new_candidate = (new_seq, (ht, ct), updated_score)
                new_beam.append(new_candidate)
                # print(new_word_idx, print(updated_score))
            # print("len(new_beam)", len(new_beam))
           
        #select top k beams based on their corresponding scores
        new_beam = sorted(new_beam, reverse=False, key=lambda x: x[2])
        beam = new_beam[-beam_size:]
        # print("beam len", len(beam), beam[0][0], beam[0][2])
        i += 1 
    best_candidate = beam[0][0] #return best candidate based on score
    # print(best_candidate)
    return best_candidate