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
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for dataloading.")

    # max number of epochs
    parser.add_argument("--epochs", type=int, default=200, help="Number of workers used for dataloading.")

    parser.add_argument("--en_hidden", type=int, default=512, help="Encoder Hidden Units")
    
    parser.add_argument("--de_hidden", type=int, default=512, help="Decoder Hidden Units")

    parser.add_argument("--en_num_layers", type=int, default=1, help="Number of lstm layers in encoder.")
    
    parser.add_argument("--de_num_layers", type=int, default=1, help="Number of lstm layers in decoder.")    
    
    parser.add_argument("--embed_dim", type=int, default=300, help="Embeddings dimension for both encoder and decoder.")

    parser.add_argument("--device", default=torch.device('cpu'), help="Device to run on.")

    parser.add_argument("--restore", type=bool, default=False , help="Restore last saved model.")

    return parser


def convert_idx_sentence(args, val_data, output, query, index, prefix):
    query = query.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    index = index.cpu().detach().numpy()

    assert len(query) == len(output) and len(query) == len(index)
    queries = []
    preds = []
    for i in range(len(query)):
        q = val_data.loc[i, 'orig_query'] + "\t" + val_data.loc[i, 'db_id'] + "\n"

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




def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.device = device
    print(device)
    if args.model_type == "Seq2Seq":
        model = Seq2Seq(args).to(device)
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedulers = [
                optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=- 1, verbose=False),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose=False)
                ]
        scheduler =  schedulers[1]
    train_dataset = Text2SQLDataset(args.processed_data, "train")
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate)

    val_dataset = Text2SQLDataset(args.processed_data, "val")
    val_loader = DataLoader(train_dataset, batch_size = args.batch_size*16, shuffle=True, 
                            num_workers=args.num_workers*2, collate_fn=collate)
    
    loss_tracker = defaultdict(list)
    time_tracker = defaultdict(list)
    min_train_loss = 10000
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
            output, _ = model(question, query)
        #     print("output and target", output.shape, query.shape)
            output = output.reshape(-1, output.shape[2])
            query = query.reshape(-1)    
        #     print("reshaped output and target", output.shape, query.shape)
            loss = criterion(output, query)
            loss.backward()
            epoch_loss.append(loss.item()/len(query))

            optimizer.step()
            # print(query)

        avg_epoch_loss = np.mean(epoch_loss)          
        scheduler.step()
        t1 = time()
        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Prev Min Train Loss: {}".format(epoch, round((t1-t0)/60,2), avg_epoch_loss, min_train_loss))
        time_tracker['train'].append(round((t1-t0)/60,2))

        model.eval()
        val_loss = [0]
        if epoch % 10 == 0:
            print("Evaluating model on val data.")            
            prefix = "train_eval"
            if os.path.exists(os.path.join(args.result_dir, f"{prefix}_gold.txt")):
                os.remove(os.path.join(args.result_dir, f"{prefix}_gold.txt"))

            if os.path.exists(os.path.join(args.result_dir, f"{prefix}_pred.txt")):
                os.remove(os.path.join(args.result_dir, f"{prefix}_pred.txt"))
            
            for i, data in enumerate(val_loader):
                question = data['question'].to(device)
                query = data['query'].to(device)

                output, words = model(question, query)
                # loss = criterion(words, query)
                # val_loss.append(loss.item()/len(query))
                convert_idx_sentence(args, val_data, words, query, data['index'], prefix)
                if i == 2:
                    break
            
            subprocess.call(f"python3 evaluation.py --gold {args.result_dir}/{prefix}_gold.txt --pred {args.result_dir}/{prefix}_pred.txt --db ./data/database/ --table ./data/tables.json --etype all", shell=True)
        
        loss_tracker['train'].append(avg_epoch_loss)
        loss_tracker['val'].append(np.mean(val_loss))
        with open(os.path.join(args.result_dir, "loss_tracker{}.json".format(args.model_type)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        torch.save(model, os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type)))
        model_state = {
                'epoch': epoch,
                'loss' : avg_epoch_loss,
                'best_loss' : min_train_loss,
                'best_epoch': best_epoch
            }
        with open(os.path.join(args.checkpoint_dir, "latest_chkpt_status_{}.json".format(args.model_type)), "w") as outfile:
            json.dump(model_state, outfile)

        if avg_epoch_loss < min_train_loss:
            model_state = {
                'epoch': epoch,
                'loss' : avg_epoch_loss,
                'prev_best_loss' : min_train_loss,
                'prev_best_epoch': best_epoch
            }
            min_train_loss = avg_epoch_loss
            best_epoch = epoch
            shutil.copy(os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type)), 
            os.path.join(args.checkpoint_dir, "best_loss_checkpoint_{}.pth".format(args.model_type)))
            
            with open(os.path.join(args.checkpoint_dir, "best_loss_chkpt_status_{}.json".format(args.model_type)), "w") as outfile:
                json.dump(model_state, outfile)








if __name__ == "__main__":
    
    #generate parser
    parser = get_parser()
    args = parser.parse_args()

    args.data_dir = os.path.relpath(args.data_dir)
    args.processed_data = os.path.relpath(args.processed_data)
    args.checkpoint_dir = os.path.join(os.path.relpath(args.checkpoint_dir), args.model_type.lower())
    args.result_dir = os.path.join(os.path.relpath(args.result_dir), args.model_type.lower())

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    print(args)

    with open(os.path.join(args.processed_data, "encoder_idx2word.pickle"), "rb") as file:
        encoder_idx2word = pickle.load(file)

    with open(os.path.join(args.processed_data, "decoder_idx2word.pickle"), "rb") as file:
        decoder_idx2word = pickle.load(file)

    
    val_data = pd.read_excel(os.path.join(args.processed_data, "val_data.xlsx"))

    train(args)