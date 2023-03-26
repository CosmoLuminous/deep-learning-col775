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
    top_sentences = []
    i = 0
    while i < max_target_len -1:
        new_beam = []
        for sequence, (ht, ct), score in beam:
            prev_token = [sequence[-1]] #get first token for each beam
            prev_token = torch.LongTensor(prev_token).to(args.device)
            # print("DECODER INPUT SHAPES", prev_token, prev_token.shape, ht.shape, ct.shape)
            decoder_out, (ht, ct) = model.decoder(prev_token, (ht, ct)) #pass through decoder
            # print("DECODER OP SHAPES", decoder_out.shape, ht.shape, ct.shape)
            decoder_out = decoder_out.squeeze(1)

            
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
    decoded_words = torch.zeros(1, max_target_len)
    # decoded_words[1, 0] = torch.LongTensor([start_token])
    # print("max_target_len, len(best_candidate)", max_target_len, len(best_candidate))
    for t in range(max_target_len):
        decoded_words[:, t] = torch.LongTensor([best_candidate[t]])
    # print(best_candidate)
    return decoded_words

def beam_search_new(args, model, hidden, cell, start_token, end_token, batch_size, max_target_len = 80, beam_size = 3):
    words = torch.zeros(batch_size, max_target_len).to(args.device)

    for b in range(batch_size):
        en_ht = hidden[:,b,:].unsqueeze(1)
        en_ct = cell[:,b,:].unsqueeze(1)

        beam = [([start_token], (en_ht, en_ct), 0)]
        # print("Beam Search input hidden and cell shape, start and end tokens", en_ht.shape, en_ct.shape, start_token, end_token)
        top_sentences = []
        i = 0
        while i < max_target_len -1:
            new_beam = []
            for sequence, (ht, ct), score in beam:
                prev_token = [sequence[-1]] #get first token for each beam
                prev_token = torch.LongTensor(prev_token).to(args.device)
                # print("DECODER INPUT SHAPES", prev_token, prev_token.shape, ht.shape, ct.shape)
                decoder_out, (ht, ct) = model.decoder(prev_token, (ht, ct)) #pass through decoder
                # print("DECODER OP SHAPES", decoder_out.shape, ht.shape, ct.shape)
                decoder_out = decoder_out.squeeze(1)

                
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
        decoded_words = torch.zeros(1, max_target_len)
        # decoded_words[1, 0] = torch.LongTensor([start_token])
        # print("max_target_len, len(best_candidate)", max_target_len, len(best_candidate))
        for t in range(max_target_len):
            words[b,t] = torch.LongTensor([best_candidate[t]]).to(args.device)
    # print(best_candidate)
    return words

def beam_search_bak(args, model, en_ht, en_ct, start_token, end_token, target_vocab_size, max_target_len = 80, beam_size = 3):
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
            decoder_out, (ht, ct) = model.decoder(prev_token, (ht, ct)) #pass through decoder
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

