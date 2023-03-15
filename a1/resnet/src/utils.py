import argparse
import os
import numpy as np
import torch
import json
from src.resnet import ResNet
import matplotlib.pyplot as plt
from glob import glob

norm_type_mappings = {"torch_bn": "Pytorch Batch Normalization", "nn": "No Normalization", 
"bn": "Custom Batch Normalization", "in": "Instance Normalization","ln": "Layer Normalization", "bin": "Batch Instance Normalization",
"gn": "Group Normalization"}

def load_checkpoint(args, model_type):

    if model_type == "best":
        model_name = os.path.join(args.checkpoint_dir, "best_val_checkpoint_{}.pth".format(args.norm_type))
    else:
        model_name = os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.norm_type))

    assert os.path.isfile(model_name), f"Model path/name invalid: {model_name}"
    
    # model_dict = torch.load(model_name)
    # net = ResNet(norm_type = args.norm_type) 
    # net.load_state_dict(model_dict['net'])
    net = torch.load(model_name)
    with open(os.path.join(args.checkpoint_dir, "training_progress_{}.json".format(args.norm_type)), "r") as file:
        model_dict = json.load(file)
    print(f"\n|--------- Model Load Success. Trained Epoch: {str(model_dict['epoch'])}, Val Accu: {str(model_dict['best_accu'])}%")

    return net

def plot_train_val_stats(args, date_time):
    loss_file = os.path.join(args.result_dir, "loss_tracker_{}_{}.json".format(args.norm_type, date_time))
    accu_file = os.path.join(args.result_dir, "accuracy_tracker_{}_{}.json".format(args.norm_type, date_time))

    with open(loss_file, "r+") as file:
        loss_dict = json.load(file)
    
    with open(accu_file, "r+") as file:
        accu_dict = json.load(file)
    best_accu_epoch = np.argmax(accu_dict['val'])
    
    fig = plt.figure()
    plt.plot(list(range(len(loss_dict['train']))), loss_dict['train'], c="tab:green", label="Train Loss")
    plt.plot(list(range(len(loss_dict['val']))), loss_dict['val'], c="tab:orange", label="Val Loss")
    plt.axvline(best_accu_epoch, linestyle="dotted", label = f"Early Stopping (epoch={best_accu_epoch})")
    plt.title(f"Train & Val Loss with for {norm_type_mappings[args.norm_type]}")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss values")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_val_loss.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_val_loss.pdf"), dpi= 300, pad_inches=0.1)


    fig = plt.figure()
    plt.plot(list(range(len(accu_dict['train']))), accu_dict['train'], c="tab:green", label="Train Accuracy")
    plt.plot(list(range(len(accu_dict['val']))), accu_dict['val'], c="tab:orange", label="Val Accuracy")
    plt.axvline(best_accu_epoch, linestyle="dotted", label = f"Early Stopping (epoch={best_accu_epoch})")
    plt.title(f"Train & Val Loss with for {norm_type_mappings[args.norm_type]}")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_val_accu.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_val_accu.pdf"), dpi= 300, pad_inches=0.1)

    return


def plot_quantiles(args, date_time):
    quantile_file = os.path.join(args.result_dir, "ft_quantile_tracker_{}_{}.json".format(args.norm_type, date_time))
    with open(quantile_file, "r+") as file:
        quantile_dict = json.load(file)

    fig = plt.figure()
    plt.plot(list(range(100)), quantile_dict['1'], c="tab:blue", label="1$^{st}$ Quantile")
    plt.plot(list(range(100)), quantile_dict['20'], c="tab:orange", label="20$^{th}$ Quantile")
    plt.plot(list(range(100)), quantile_dict['80'], c="tab:green", label="80$^{th}$ Quantile")
    plt.plot(list(range(100)), quantile_dict['99'], c="tab:purple", label="99$^{th}$ Quantile")
    plt.title(f"Quantile plot for {norm_type_mappings[args.norm_type]}")
    plt.xlabel("# Epochs")
    plt.ylabel("Feature Value")
    plt.legend()
    
    fig.savefig(os.path.join(args.result_dir, "ft_quantile_plot.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "ft_quantile_plot.pdf"), dpi= 300, pad_inches=0.1)

    return


def plot_loss_comparison(args):

    fig = plt.figure()
    path = os.path.join(args.result_dir, "**/loss_tracker*.json")
    json_list = glob(path)
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['train'], label=norm_type.upper())
        
    plt.title(f"Train Loss variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss Values")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_loss_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_loss_all.pdf"), dpi= 300, pad_inches=0.1)

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/loss_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            # print(norm_type)
            plt.plot(list(range(100)), json_dict['val'], label=norm_type.upper())
        
    plt.title(f"Validation Loss variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss Values")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "val_loss_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "val_loss_all.pdf"), dpi= 300, pad_inches=0.1)

    return

def plot_accu_comparison(args):

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/accuracy_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['train'], label=norm_type.upper())
        
    plt.title(f"Train Accuracy variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_accu_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_accu_all.pdf"), dpi= 300, pad_inches=0.1)

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/accuracy_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['val'], label=norm_type.upper())
        
    plt.title(f"Validation Accuracy variation with Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "val_accu_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "val_accu_all.pdf"), dpi= 300, pad_inches=0.1)

    return

def plot_time_comparison(args):

    fig = plt.figure()
    json_list = glob(os.path.join(args.result_dir, "**/time_tracker*.json"))
    json_list.sort()
    for file in json_list:
        with open(file, "r+") as f:
            json_dict = json.load(f)
        
        norm_type = file.split("/")[-2]
        if norm_type != "torch_bn":
            print(norm_type)
            plt.plot(list(range(100)), json_dict['train'], label=norm_type.upper())
        
    plt.title(f"Training Time v/s Epochs")
    plt.xlabel("# Epochs")
    plt.ylabel("Time (Mins)")
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, "train_time_all.png"), dpi= 300, pad_inches=0.1, format="png")
    fig.savefig(os.path.join(args.result_dir, "train_time_all.pdf"), dpi= 300, pad_inches=0.1)

    return