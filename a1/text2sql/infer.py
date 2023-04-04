import json
from collections import Counter, defaultdict
import pickle
import pandas as pd
import subprocess
import argparse
import os
import json
import pickle
import shutil
from time import time
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from beam_search import beam_search, greedy_decoder, beam_search_attn_decoder
from dataset import Text2SQLDataset, collate, Text2SQLBertDataset, collate_bert
from utils import *
from beam_search import beam_search, greedy_decoder, beam_search_attn_decoder

def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Text2SQL Inference")

    # path to data files.
    parser.add_argument("--test_data_file", type=str, help="Test data file path.")

    parser.add_argument("--output_file", type=str, help="Output file path.")

    parser.add_argument("--model_file", type=str, help="Model File Path.")



    # model type
    parser.add_argument("--model_type", type=str, default="lstm_lstm", help="Select the model you want to run from [lstm_lstm | lstm_lstm_attn | bert_lstm_attn_frozen | bert_lstm_attn_tuned].")

    # path to data files.
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory.")

    # path to result files.
    parser.add_argument("--result_dir", type=str, default="./results", help="Path to dataset directory.")

    # path to model checkpoints.
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to model checkpoints.")
    
    # path to model checkpoints.
    parser.add_argument("--processed_data", type=str, default="./processed_data", help="Path to processed data.")

    # batch size training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to be used during training.")

    # number of workers for dataloader
    parser.add_argument("--num_workers", type=int, default=24, help="Number of workers used for dataloading.")

    # max number of epochs
    parser.add_argument("--epochs", type=int, default=200, help="Number of workers used for dataloading.")

    parser.add_argument("--en_hidden", type=int, default=512, help="Encoder Hidden Units")
    
    parser.add_argument("--de_hidden", type=int, default=512, help="Decoder Hidden Units")

    parser.add_argument("--en_num_layers", type=int, default=1, help="Number of lstm layers in encoder.")
    
    parser.add_argument("--de_num_layers", type=int, default=1, help="Number of lstm layers in decoder.")    
    
    parser.add_argument("--embed_dim", type=int, default=300, help="Embeddings dimension for both encoder and decoder.")

    parser.add_argument("--device", default=torch.device('cpu'), help="Device to run on.")

    parser.add_argument("--restore", type=bool, default=False , help="Restore last saved model.")

    parser.add_argument("--search_type", type=str, default="beam", help="Type of {greedy, beam} search to perform on decoder.")

    parser.add_argument("--beam_size", type=int, default=1, help="Beam size to be used during beam search decoding.")

    parser.add_argument("--bert_tune_layers", type=int, default=-1, help="Number of layers to finetune for bert encoder.")
    
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for the model.")

    return parser


def process_data(file_path, output_path, prefix = 'train'):
    
    data_points = []    
    data_file = pd.read_csv(file_path)

    for idx, dp in data_file.iterrows():
        question = dp["question"]
        ques_tokens = " ".join(tokenize_question(question))
        query = dp["query"]
        query_tokens = " ".join(tokenize_query(query))
        db_id = dp['db_id']

        data_points.append([db_id, ques_tokens, query_tokens, query.lower().replace('\t', ' ')])
    
    df = pd.DataFrame(data_points, columns=["db_id", "question", "query", "orig_query"])
    file_name = os.path.join(output_path, "{}_data.xlsx".format(prefix))
    df.to_excel(file_name, index=False)

    return df

def load_test_checkpoint(model_name):

    # if chkpt == "best":
    #     model_name = os.path.join(args.checkpoint_dir, "best_loss_checkpoint_{}.pth".format(args.model_type))
    #     status_file = os.path.join(args.checkpoint_dir, "best_loss_chkpt_status_{}.json".format(args.model_type))
    # else:
    #     model_name = os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type))
    #     status_file = os.path.join(args.checkpoint_dir, "latest_chkpt_status_{}.json".format(args.model_type))

    assert os.path.isfile(model_name), f"Model path/name invalid: {model_name}"
    
    net = torch.load(model_name)
    
    print(f"\n|--------- Model Load Success.")

    return net

def get_eval_stats(result_file):
    with open(result_file, "r") as file:
        data = file.readlines()
    try:
        a,b,c,d = data[0].split()[-1], data[1].split()[-1], data[3].split()[-1], data[6].split()[-1]
    except:        
        print("Error in calculating performance. Returning -1")
        a,b,c,d = (-1, -1, -1, -1)
    return a,b,c,d

def convert_idx_sentence(args, output, og_query, db_id, prefix="test"):
    output = output.cpu().detach().numpy()
    og_query = list(og_query)
    db_id = list(db_id)
    assert len(og_query) == len(output) and len(og_query) == len(db_id)
    queries = []
    preds = []
    for i in range(len(og_query)):
        q = og_query[i] + "\t" + db_id[i] + "\n"

        p = []
        for idx in output[i]:
            w = decoder_idx2word[idx]
            if w == "<eos>":
                break
            else:
                p.append(w)
        
        p.append("\n")
        queries.append(q)
        preds.append(" ".join(p[1:]))

    with open(os.path.join("./", f"{prefix}_gold.txt"), "a") as file:
        file.writelines(queries)
    
    with open(os.path.join("./", f"{prefix}_pred.txt"), "a") as file:
        file.writelines(preds)

    return
def model_eval(args, prefix="test", model=None, de_word2idx = None, val_loader=None):
    print(f"\n------------------------------ Running inference for: {args.model_type} ------------------------------")
    if val_loader == None:
        if args.model_type == "bert_lstm_attn_frozen" or args.model_type == "bert_lstm_attn_tuned":
            val_dataset = Text2SQLBertDataset(args.processed_data, "test")
            val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, 
            num_workers=args.num_workers, collate_fn=collate_bert)
        else:
            val_dataset = Text2SQLDataset(args.processed_data, "test")

            val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, 
            num_workers=args.num_workers, collate_fn=collate)

        de_word2idx = val_dataset.de_word2idx

    if model == None:
        print("loading saved model...",args.model_file)
        model = load_test_checkpoint(args.model_file)
        model = model.to(args.device)
        model.eval()
    else:
        print("using existing model...")

    print("Evaluating model on val data.")
    prefix = args.model_type + "_" + prefix
    gold_file = os.path.join("./", f"{prefix}_gold.txt")
    pred_file = os.path.join("./", f"{prefix}_pred.txt")
    fine_path = args.output_file.split("/")
    new_out = prefix + "_" + fine_path[-1]
    
    results_file = os.path.join("./", args.output_file.replace(fine_path[-1], new_out).replace("csv", "txt"))
    
    if os.path.exists(gold_file):
        os.remove(gold_file)

    if os.path.exists(pred_file):
        os.remove(pred_file)    
    
    if os.path.exists(results_file):
        os.remove(results_file)
    
    
    start_token = de_word2idx["<sos>"]
    end_token = de_word2idx["<eos>"]
    target_vocab_size = len(de_word2idx)
    decoder_hidden_units = model.decoder_hidden_units

    for i, data in enumerate(val_loader):
        
        question = data['question'].to(device)
        query = data['query'].to(device)
        og_query = data['og_query']
        db_id = data['db_id']

        batch_size = question.shape[0]
        max_target_len = query.shape[1]

        words = torch.zeros(batch_size, max_target_len).to(args.device)
        if args.model_type == "lstm_lstm" or  args.model_type == "lstm_lstm_attn":            
            encoder_out, (hidden, cell) = model.encoder(question)
        else:
            ques_attn_mask = data['ques_attn_mask'].to(device)
            encoder_out = model.encoder(question, ques_attn_mask)
            hidden = torch.zeros(1, batch_size, decoder_hidden_units).to(args.device)
            cell = torch.zeros(1, batch_size, decoder_hidden_units).to(args.device)

        if args.search_type == "beam":
            for b in range(batch_size):
                if args.model_type == "lstm_lstm":
                    words[b,:] = beam_search(args, model, hidden[:,b,:].unsqueeze(1), cell[:,b,:].unsqueeze(1),
                    start_token, end_token, max_target_len, args.beam_size)
                else:
                    words[b,:] = beam_search_attn_decoder(args, model, encoder_out[b,:,:].unsqueeze(0), hidden[:,b,:].unsqueeze(1), cell[:,b,:].unsqueeze(1),
                    start_token, end_token, max_target_len, args.beam_size)
        else:
            words = greedy_decoder(args, model, hidden, cell, query, batch_size, target_vocab_size, max_target_len)
        
        convert_idx_sentence(args, words, og_query, db_id, prefix)

    print("Running evaluation script...")
    subprocess.call(f"python3 evaluation.py --gold {gold_file} --pred {pred_file} --db ./data/database/ --table ./data/tables.json --etype all >> {results_file}", shell=True)
    _, _, exec_accu, exact_match_accu = get_eval_stats(results_file)
    
    exec_accu = round(np.float64(exec_accu),3)
    exact_match_accu = round(np.float64(exact_match_accu), 3)
    print("|----------- Execution Accuracy = {}, Exact Match Accuracy = {} -----------|\n\n\n".format(exec_accu, exact_match_accu))

    with open(pred_file, "r") as f:
        pred_file_content = f.readlines()

    df_pred = pd.DataFrame([line.replace(" \n", "") for line in pred_file_content])
    df_pred.to_csv(args.output_file, index=False, header=False)

    return exec_accu, exact_match_accu 


if __name__ == "__main__":
    
    #generate parser
    parser = get_parser()
    args = parser.parse_args()

    args.data_dir = os.path.relpath(args.data_dir)
    args.processed_data = os.path.relpath(args.processed_data)
    if args.model_type == "bert_lstm_attn_tuned":
        extension = args.model_type.lower() + str(args.bert_tune_layers) + "_" + str(args.lr) + "_" + str(args.en_num_layers) + "_" + str(args.de_num_layers) + "_" + str(args.en_hidden) + "_" + str(args.de_hidden) + "_" + str(args.embed_dim) + "_" + str(args.batch_size) + "_" + str(args.epochs)    
    else:
        extension = args.model_type.lower() + "_" + str(args.lr) + "_" + str(args.en_num_layers) + "_" + str(args.de_num_layers) + "_" + str(args.en_hidden) + "_" + str(args.de_hidden) + "_" + str(args.embed_dim) + "_" + str(args.batch_size) + "_" + str(args.epochs)
    args.checkpoint_dir = os.path.join(os.path.relpath(args.checkpoint_dir), extension)
    args.result_dir = os.path.join(os.path.relpath(args.result_dir), extension)

    # if not os.path.isdir(args.checkpoint_dir):
    #     os.mkdir(args.checkpoint_dir)
    # if not os.path.isdir(args.result_dir):
    #     os.mkdir(args.result_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.device = device

    print(args)
    
    print("Running on device:", device)

    with open(os.path.join(args.processed_data, "encoder_idx2word.pickle"), "rb") as file:
        encoder_idx2word = pickle.load(file)

    with open(os.path.join(args.processed_data, "decoder_idx2word.pickle"), "rb") as file:
        decoder_idx2word = pickle.load(file)

    test_data = process_data(args.test_data_file, "./processed_data/", "test")
    model_eval(args)