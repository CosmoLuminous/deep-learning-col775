import json
from collections import Counter, defaultdict
import pickle
from utils import *
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class Text2SQLDataset(Dataset):
    def __init__(self, file_path, data_prefix = "train"):
        self.file_path = file_path
        self.data = pd.read_excel(os.path.join(file_path, f"{data_prefix}_data.xlsx"))
        print("Dataset Length =", len(self.data))
        with open(os.path.join(file_path, "encoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.encoder_vocab = vocab
        
        with open(os.path.join(file_path, "decoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.decoder_vocab = vocab
        
        with open(os.path.join(file_path, "encoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "encoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
            
        self.en_word2idx = word2idx
        self.en_idx2word = idx2word
        
        with open(os.path.join(file_path, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "decoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
            
        self.de_word2idx = word2idx
        self.de_idx2word = idx2word
        print("Encoder Vocab Size = {}, Decoder Vocab Size = {}".format(len(self.en_word2idx), len(self.de_word2idx)))
        
    def __len__(self):        
        return len(self.data)
    
    def __getitem__(self, idx):
#         print(idx, "\n")
        
        query = ["<sos>"] + tokenize_query(self.data.loc[idx, "query"]) + ["<eos>"]
        question =  ["<sos>"] + tokenize_question(self.data.loc[idx, "question"]) + ["<eos>"]
        # print(query)
        query = [self.de_word2idx[q] if q in self.de_word2idx else self.de_word2idx["<unk>"] for q in query]
        question = [self.en_word2idx[q] if q in self.en_word2idx else self.en_word2idx["<unk>"] for q in question]
        # print(query)
        og_query = self.data.loc[idx, "orig_query"]
        db_id =  self.data.loc[idx, 'db_id']
        sample = {'question': question, 'query': query, 'db_id': db_id, 'og_query': og_query}
        
            
        return sample
    
def collate(batch):
    
    max_len_ques = max([len(sample['question']) for sample in batch])
    max_len_query = max([len(sample['query']) for sample in batch])
    
    ques_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_ques = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    
    query_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_query = torch.zeros((len(batch), max_len_query), dtype=torch.long)
    indexes = torch.zeros(len(batch), dtype=torch.long)
    og_query = list()
    db_id = list()

    for idx in range(len(batch)):
        
        query = batch[idx]['query']
        question = batch[idx]['question']
        
        ques_len = len(question)
        query_len = len(query)
        ques_lens[idx] = ques_len
        query_lens[idx] = query_len
        # indexes[idx] = batch[idx]['index']
        
        padded_ques[idx, :ques_len] = torch.LongTensor(question)
        padded_query[idx, :query_len] = torch.LongTensor(query)
        og_query.append(batch[idx]['og_query'])
        db_id.append(batch[idx]['db_id'])
        
    return {'question': padded_ques, 'query': padded_query, 'ques_lens': query_lens, 'query_lens': query_lens, 'db_id': db_id, 'og_query': og_query}


class Text2SQLBertDataset(Dataset):
    def __init__(self, file_path, data_prefix = "train"):
        self.file_path = file_path
        self.data = pd.read_excel(os.path.join(file_path, f"{data_prefix}_data.xlsx"))
        print("Dataset Length =", len(self.data))
        # with open(os.path.join(file_path, "encoder.vocab"), "r") as file:
        #     vocab = file.readlines()
        # self.encoder_vocab = vocab
        
        with open(os.path.join(file_path, "decoder.vocab"), "r") as file:
            vocab = file.readlines()
        self.decoder_vocab = vocab
        
        # with open(os.path.join(file_path, "encoder_word2idx.pickle"), "rb") as file:
        #     word2idx = pickle.load(file)
        # with open(os.path.join(file_path, "encoder_idx2word.pickle"), "rb") as file:
        #     idx2word = pickle.load(file)
            
        # self.en_word2idx = word2idx
        # self.en_idx2word = idx2word
        
        with open(os.path.join(file_path, "decoder_word2idx.pickle"), "rb") as file:
            word2idx = pickle.load(file)
        with open(os.path.join(file_path, "decoder_idx2word.pickle"), "rb") as file:
            idx2word = pickle.load(file)
            
        self.de_word2idx = word2idx
        self.de_idx2word = idx2word

        self.en_tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")


        print("Encoder Vocab Size = , Decoder Vocab Size = {}".format( len(self.de_word2idx)))
        
    def __len__(self):        
        return len(self.data)
    
    def __getitem__(self, idx):
#         print(idx, "\n")
        
        query = ["<sos>"] + tokenize_query(self.data.loc[idx, "query"]) + ["<eos>"]
        
        # print(query)
        query = [self.de_word2idx[q] if q in self.de_word2idx else self.de_word2idx["<unk>"] for q in query]
        
        # question = self.data.loc[idx, "question"]
        question = self.en_tokenizer.encode(self.data.loc[idx, "question"])
        # print(query)
        og_query = self.data.loc[idx, "orig_query"]
        db_id =  self.data.loc[idx, 'db_id']
        sample = {'question': question, 'query': query, 'db_id': db_id, 'og_query': og_query}
        
            
        return sample

def collate_bert(batch):
    
    max_len_ques = max([len(sample['question']) for sample in batch])
    max_len_query = max([len(sample['query']) for sample in batch])
    
    ques_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_ques = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    ques_attn_mask = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    
    query_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_query = torch.zeros((len(batch), max_len_query), dtype=torch.long)
    indexes = torch.zeros(len(batch), dtype=torch.long)
    og_query = list()
    db_id = list()

    for idx in range(len(batch)):
        
        query = batch[idx]['query']
        question = batch[idx]['question']
        
        ques_len = len(question)
        query_len = len(query)
        ques_lens[idx] = ques_len
        query_lens[idx] = query_len
        # indexes[idx] = batch[idx]['index']
        
        padded_ques[idx, :ques_len] = torch.LongTensor(question)
        ques_attn_mask[idx, :ques_len] = torch.ones((1, ques_len), dtype=torch.long)

        padded_query[idx, :query_len] = torch.LongTensor(query)
        
        og_query.append(batch[idx]['og_query'])
        db_id.append(batch[idx]['db_id'])
        
    return {'question': padded_ques, 'ques_attn_mask': ques_attn_mask, 'query': padded_query, 'ques_lens': query_lens, 'query_lens': query_lens, 'db_id': db_id, 'og_query': og_query}
