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
from utils import *


def beam_search(args, model, en_ht, en_ct, start_token, end_token, max_target_len = 80, beam_size = 3):
    beam = [([start_token], (en_ht, en_ct), 0)]
    # print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)

    i = 0
    while i < max_target_len -1:
        new_beam = []
        for sequence, (ht, ct), score in beam:
            prev_token = [sequence[-1]] #get first token for each beam
            prev_token = torch.LongTensor(prev_token).to(args.device)

            decoder_out, (ht, ct) = model.decoder(prev_token, (ht, ct)) #pass through decoder

            decoder_out = decoder_out.squeeze(1)
            top_info = decoder_out.topk(beam_size, dim=1) #get top k=beam_size possible word indices and their values
            top_vals, top_inds = top_info

            for j in range(beam_size):
                new_word_idx = top_inds[0][j]                
                new_seq = sequence + [new_word_idx.item()]
                new_word_prob = torch.log(top_vals[0][j])
                updated_score = score - new_word_prob
                new_candidate = (new_seq, (ht, ct), updated_score)
                new_beam.append(new_candidate)

        # new_beam = sorted(new_beam, reverse=False, key=lambda x: x[2])
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_size]
        i += 1

    best_candidate = beam[0][0] #return best candidate based on score
    decoded_words = torch.zeros(1, max_target_len)

    for t in range(max_target_len):
        decoded_words[:, t] = torch.LongTensor([best_candidate[t]])
    
    return decoded_words



# def beam_search_bak(args, model, en_ht, en_ct, start_token, end_token, max_target_len = 80, beam_size = 3):
#     beam = [([start_token], (en_ht, en_ct), 0)]
#     # print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)

#     i = 0
#     while i < max_target_len -1:
#         new_beam = []
#         for sequence, (ht, ct), score in beam:
#             prev_token = [sequence[-1]] #get first token for each beam
#             prev_token = torch.LongTensor(prev_token).to(args.device)

#             decoder_out, (ht, ct) = model.decoder(prev_token, (ht, ct)) #pass through decoder

#             decoder_out = decoder_out.squeeze(1)
#             top_info = decoder_out.topk(beam_size, dim=1) #get top k=beam_size possible word indices and their values
#             top_vals, top_inds = top_info

#             for j in range(beam_size):
#                 new_word_idx = top_inds[0][j]                
#                 new_seq = sequence + [new_word_idx.item()]
#                 new_word_prob = torch.log(top_vals[0][j])
#                 updated_score = score - new_word_prob
#                 new_candidate = (new_seq, (ht, ct), updated_score)
#                 new_beam.append(new_candidate)

#         # new_beam = sorted(new_beam, reverse=True, key=lambda x: x[2])
#         new_beam.sort(key=lambda x: x[2])
#         beam = new_beam[:beam_size]
#         i += 1

#     best_candidate = beam[0][0] #return best candidate based on score
#     decoded_words = torch.zeros(1, max_target_len)

#     for t in range(max_target_len):
#         decoded_words[:, t] = torch.LongTensor([best_candidate[t]])
    
#     return decoded_words

def beam_search_attn_decoder(args, model, encoder_out, en_ht, en_ct, start_token, end_token, max_target_len = 80, beam_size = 3):
    beam = [([start_token], (en_ht, en_ct), 0)]
    # print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)

    i = 0
    while i < max_target_len -1:
        new_beam = []
        for sequence, (ht, ct), score in beam:
            prev_token = [sequence[-1]] #get first token for each beam
            prev_token = torch.LongTensor(prev_token).to(args.device)

            decoder_out, (ht, ct) = model.decoder(prev_token, (ht, ct), encoder_out) #pass through decoder

            decoder_out = decoder_out.squeeze(1)
            top_info = decoder_out.topk(beam_size, dim=1) #get top k=beam_size possible word indices and their values
            top_vals, top_inds = top_info

            for j in range(beam_size):
                new_word_idx = top_inds[0][j]                
                new_seq = sequence + [new_word_idx.item()]
                new_word_prob = torch.log(top_vals[0][j])
                updated_score = score - new_word_prob
                new_candidate = (new_seq, (ht, ct), updated_score)
                new_beam.append(new_candidate)

        # new_beam = sorted(new_beam, reverse=True, key=lambda x: x[2])
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_size]
        i += 1

    best_candidate = beam[0][0] #return best candidate based on score
    decoded_words = torch.zeros(1, max_target_len)

    for t in range(max_target_len):
        decoded_words[:, t] = torch.LongTensor([best_candidate[t]])
    
    return decoded_words


def greedy_decoder(args, model, hidden, cell, query, batch_size, target_vocab_size, max_target_len = 80):

    outputs = torch.zeros(batch_size, max_target_len, target_vocab_size).to(args.device)
    words = torch.zeros(batch_size, max_target_len).to(args.device)   

    x = query[:,0]            
    words[:, 0] = query[:,0]
    for t in range(1, max_target_len):
        output, (hidden, cell) = model.decoder(x, (hidden, cell))
        output = output.squeeze(1)
        outputs[:,t,:] = output
        x = output.argmax(dim=1)
        words[:, t] = x
    return words