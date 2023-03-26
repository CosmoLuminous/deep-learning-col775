import argparse
import os
import subprocess
import json
import pickle
import shutil
from time import time
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn

from text2sql import Seq2Seq
from dataset import Text2SQLDataset, collate
from utils import *
from beam_search import beam_search

def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Text2SQL")
    
    # model type
    parser.add_argument("--model_type", type=str, default="Seq2Seq", help="Select the model you want to run from [Seq2Seq, Seq2SeqAttn].")

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
    parser.add_argument("--epochs", type=int, default=300, help="Number of workers used for dataloading.")

    parser.add_argument("--en_hidden", type=int, default=512, help="Encoder Hidden Units")
    
    parser.add_argument("--de_hidden", type=int, default=512, help="Decoder Hidden Units")

    parser.add_argument("--en_num_layers", type=int, default=1, help="Number of lstm layers in encoder.")
    
    parser.add_argument("--de_num_layers", type=int, default=1, help="Number of lstm layers in decoder.")    
    
    parser.add_argument("--embed_dim", type=int, default=300, help="Embeddings dimension for both encoder and decoder.")

    parser.add_argument("--device", default=torch.device('cpu'), help="Device to run on.")

    parser.add_argument("--restore", type=bool, default=False , help="Restore last saved model.")

    parser.add_argument("--search_type", type=str, default="greedy", help="Type of {greedy, beam} search to perform on decoder.")

    return parser


def convert_idx_sentence(args, output, og_query, db_id, prefix):
    # query = query.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    # index = index.cpu().detach().numpy()
    # index = np.sort(index)
    og_query = list(og_query)
    db_id = list(db_id)

    # print(index)
    # print(len(og_query), len(output), len(og_query), len(db_id))
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

    with open(os.path.join(args.result_dir, f"{prefix}_gold.txt"), "a") as file:
        file.writelines(queries)
    
    with open(os.path.join(args.result_dir, f"{prefix}_pred.txt"), "a") as file:
        file.writelines(preds)

    return

def get_eval_stats(args, prefix):
    with open(f"{args.result_dir}/{prefix}_results.txt", "r") as file:
        data = file.readlines()
    try:
        a,b,c,d = data[0].split()[-1], data[1].split()[-1], data[3].split()[-1], data[6].split()[-1]
    except:        
        print("Error in calculating performance. Returning -1")
        a,b,c,d = (-1, -1, -1, -1)
    return a,b,c,d



def train(args):
    if args.model_type == "Seq2Seq":
        model = Seq2Seq(args).to(device)
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedulers = [
                optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch= -1, verbose=False),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose=False)
                ]
        scheduler =  schedulers[1]
    train_dataset = Text2SQLDataset(args.processed_data, "train")
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate)

    val_dataset = Text2SQLDataset(args.processed_data, "val")
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=collate)
    
    loss_tracker = defaultdict(list)
    time_tracker = defaultdict(list)
    val_accuracy_tracker = defaultdict(list)
    min_train_loss = 10000
    best_exec_accu = -1
    best_match_accu = -1
    best_epoch = 0
    t0 = time()
    
    for epoch in range(args.epochs):
        model.train()
        print("\n\n\n|===================================== Epoch: {} =====================================|".format(epoch))
        epoch_loss = []
        for i, data in enumerate(train_loader):
        #     print(data['question'].shape, data['query'].shape, data['ques_lens'].shape, data['query_lens'].shape)
            optimizer.zero_grad()
            question = data['question'].to(device)
            query = data['query'].to(device)
            output = model(question, query)
        #     print("output and target", output.shape, query.shape)
            output = output.reshape(-1, output.shape[2])
            query = query.reshape(-1)    
        #     print("reshaped output and target", output.shape, query.shape)
            loss = criterion(output, query)
            loss.backward()
            epoch_loss.append(loss.item())

            optimizer.step()
            # print(query)
        
        scheduler.step()
        t1 = time()

        model.eval()
        val_loss = []
        if epoch % 1 == 0:
            print("Evaluating model on val data.")            
            prefix = "train_eval"
            # if os.path.exists(os.path.join(args.result_dir, f"{prefix}_gold.txt")):
            #     os.remove(os.path.join(args.result_dir, f"{prefix}_gold.txt"))

            # if os.path.exists(os.path.join(args.result_dir, f"{prefix}_pred.txt")):
            #     os.remove(os.path.join(args.result_dir, f"{prefix}_pred.txt"))
            
            for i, data in enumerate(val_loader):
                question = data['question'].to(device)
                query = data['query'].to(device)
                # og_query = data['og_query']
                # db_id = data['db_id']

                output = model(question, query)
                output = output.reshape(-1, output.shape[2])
                query = query.reshape(-1)    
                loss = criterion(output, query)
                val_loss.append(loss.item())
                # convert_idx_sentence(args, words, og_query, db_id, prefix)
            if epoch % 5 == 0:
                exec_accu, exact_match_accu = model_eval(args, prefix, model, val_dataset)
                # exec_accu = round(np.float64(exec_accu),3)
                # exact_match_accu = round(np.float64(exact_match_accu), 3)
            val_accuracy_tracker["exec_accu"].append(exec_accu)
            val_accuracy_tracker["exact_match_accu"].append(exact_match_accu)
            
        avg_val_loss = np.mean(val_loss)
        avg_epoch_loss = np.mean(epoch_loss)  
        
        loss_tracker['train'].append(avg_epoch_loss)
        loss_tracker['val'].append(avg_val_loss)

        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Val Loss: {}, Execution Accuracy = {}, Exact Match Accuracy = {}".format(epoch, round((t1-t0)/60,2), avg_epoch_loss, avg_val_loss, val_accuracy_tracker["exec_accu"][-1], val_accuracy_tracker["exact_match_accu"][-1]))
        time_tracker['train'].append(round((t1-t0)/60,2))

        with open(os.path.join(args.result_dir, "loss_tracker{}.json".format(args.model_type)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        with open(os.path.join(args.result_dir, "val_accuracy_tracker{}.json".format(args.model_type)), "w") as outfile:
            json.dump(val_accuracy_tracker, outfile)

        torch.save(model, os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type)))
        model_state = {
                'epoch': epoch,
                'train_loss' : avg_epoch_loss,
                'val_loss' : avg_val_loss,
                'prev_best_accu' : best_match_accu,
                'prev_best_epoch': best_epoch
            }
        with open(os.path.join(args.checkpoint_dir, "latest_chkpt_status_{}.json".format(args.model_type)), "w") as outfile:
            json.dump(model_state, outfile)

        if exact_match_accu > best_match_accu:
            model_state = {
                'epoch': epoch,
                'train_loss' : avg_epoch_loss,
                'val_loss' : avg_val_loss,
                'prev_best_accu' : best_match_accu,
                'prev_best_epoch': best_epoch
            }
            min_train_loss = avg_epoch_loss
            best_epoch = epoch
            shutil.copy(os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type)), 
            os.path.join(args.checkpoint_dir, "best_loss_checkpoint_{}.pth".format(args.model_type)))
            
            with open(os.path.join(args.checkpoint_dir, "best_loss_chkpt_status_{}.json".format(args.model_type)), "w") as outfile:
                json.dump(model_state, outfile)

    return

def model_eval(args, prefix, model=None, val_dataset=None):
    if val_dataset == None:
        val_dataset = Text2SQLDataset(args.processed_data, "val")

        val_loader = DataLoader(val_dataset, batch_size = args.batch_size*16, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=collate)
    
    if model == None:
        print("loading saved model...")
        model = load_checkpoint(args, "best")
        model = model.to(args.device)
        model.eval()
    else:
        print("using existing model...")
    
    print("Evaluating model on val data.")            
    
    if os.path.exists(os.path.join(args.result_dir, f"{prefix}_gold.txt")):
        os.remove(os.path.join(args.result_dir, f"{prefix}_gold.txt"))

    if os.path.exists(os.path.join(args.result_dir, f"{prefix}_pred.txt")):
        os.remove(os.path.join(args.result_dir, f"{prefix}_pred.txt"))

    target_vocab = val_dataset.de_word2idx
    start_token = target_vocab["<sos>"]
    end_token = target_vocab["<eos>"]
    target_vocab_size = len(target_vocab)
    print("target vocab size", target_vocab_size)

    for i, data in enumerate(val_loader):
        # print("entered val loader loop.")
        question = data['question'].to(device)
        query = data['query'].to(device)
        og_query = data['og_query']
        db_id = data['db_id']
        
        # output, words = model(question, query)
        # loss = criterion(words, query)
        # val_loss.append(loss.item()/len(query))
        
        batch_size = question.shape[0]
        max_target_len = query.shape[1]
        
        # print(f"EVAL BATCH {i}, SIZE {batch_size}")
        
        words = torch.zeros(batch_size, max_target_len).to(args.device)

        _, (hidden, cell) = model.encoder(question)
        
        for b in range(batch_size):
            words[b,:] = beam_search(args, model, hidden[:,b,:].unsqueeze(1), cell[:,b,:].unsqueeze(1), start_token, end_token, max_target_len, 3)
        
        convert_idx_sentence(args, words, og_query, db_id, prefix)
        
    print("Running evaluation script...")
    if os.path.exists(f"{args.result_dir}/{prefix}_results.txt"):
        os.remove(f"{args.result_dir}/{prefix}_results.txt")

    subprocess.call(f"python3 evaluation.py --gold {args.result_dir}/{prefix}_gold.txt --pred {args.result_dir}/{prefix}_pred.txt --db ./data/database/ --table ./data/tables.json --etype all >> {args.result_dir}/{prefix}_results.txt", shell=True)
    _, _, exec_accu, exact_match_accu = get_eval_stats(args, prefix)
    
    exec_accu = round(np.float64(exec_accu),3)
    exact_match_accu = round(np.float64(exact_match_accu), 3)
    print("Execution Accuracy = {}, Exact Match Accuracy = {}".format(exec_accu, exact_match_accu))

    return exec_accu, exact_match_accu 









if __name__ == "__main__":
    
    #generate parser
    parser = get_parser()
    args = parser.parse_args()

    args.data_dir = os.path.relpath(args.data_dir)
    args.processed_data = os.path.relpath(args.processed_data)
    extension = args.model_type.lower() + "_" + str(args.en_num_layers) + "_" + str(args.de_num_layers) + "_" + str(args.en_hidden) + "_" + str(args.de_hidden) + "_" + str(args.embed_dim) + "_" + str(args.batch_size) + "_" + str(args.epochs)
    args.checkpoint_dir = os.path.join(os.path.relpath(args.checkpoint_dir), extension)
    args.result_dir = os.path.join(os.path.relpath(args.result_dir), extension)

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.device = device

    print(args)
    
    print("Running on device:", device)

    with open(os.path.join(args.processed_data, "encoder_idx2word.pickle"), "rb") as file:
        encoder_idx2word = pickle.load(file)

    with open(os.path.join(args.processed_data, "decoder_idx2word.pickle"), "rb") as file:
        decoder_idx2word = pickle.load(file)

    
    val_data = pd.read_excel(os.path.join(args.processed_data, "val_data.xlsx"))

    train(args)
    model_eval(args, 'test_eval')