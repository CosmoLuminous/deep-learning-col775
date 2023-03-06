import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import json 
from time import time
from tqdm import tqdm
from datetime import datetime

from src.resnet import ResNet
from src.utils import load_checkpoint

def train(args):

    if args.aug:

        train_transforms = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])
    
    else:

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ])


    dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transforms)
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [40000, 10000])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = args.batch_size*4, shuffle = False, num_workers = args.num_workers
    )
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))])
    
    test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = args.batch_size*4, shuffle = False, num_workers = args.num_workers
    )

    print("Train Size = {}, Val Size = {}, Test Size = {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    

    net = ResNet(norm_type = args.norm_type)
    print(net)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) #Check for self.epochs param
    
    loss_tracker = defaultdict(list)
    accuracy_tracker = defaultdict(list)
    best_accuracy = -1
    best_accu_epoch = -1

    print("\n\n---------------------------- MODEL TRAINING BEGINS ----------------------------")
        
    t0 = time()
    for epoch in range(args.epochs):
        print("\n#------------------ Epoch: %d ------------------#" % epoch)

        train_loss = []
        correct_pred = 0
        total_samples = 0
        
        net.train()
        for idx, batch in enumerate(train_loader):
                
            optimizer.zero_grad()
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            
            loss = criterion(outputs, labels)
            
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            verdict = torch.eq(pred, labels)
            correct_pred += verdict.sum().item()
            total_samples += labels.size(0)

        loss_tracker["train"].append(np.mean(train_loss))
        accuracy_tracker["train"].append(round(correct_pred/total_samples*100, 2))

        scheduler.step()

        net.eval()
        correct_pred = 0
        total_samples = 0
        val_loss = []
        for idx, batch in enumerate(val_loader):
            
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)        
            val_loss.append(loss.item())

            _, pred = outputs.max(1)
            verdict = torch.eq(pred, labels)
            correct_pred += verdict.sum().item()
            total_samples += labels.size(0)
        
        loss_tracker["val"].append(np.mean(val_loss))
        val_accuracy = round(correct_pred/total_samples*100, 2)
        accuracy_tracker["val"].append(val_accuracy)

        t1 = time()

        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Train Accuracy: {}%, Validation Loss: {}, Validation Accuracy: {}%".format(epoch, round((t1-t0)/60,2), loss_tracker["train"][-1], accuracy_tracker["train"][-1], loss_tracker["val"][-1], accuracy_tracker["val"][-1]))

        model_state = {
                'net': net.state_dict(),
                'accu': val_accuracy,
                'epoch': epoch,
                'best_accu': best_accuracy,
                'best_accu_epoch': best_accu_epoch
            }

        print("Epoch: {}, Saving Model Checkpoint: {}".format(epoch, now.strftime("%d-%m-%y %H:%M")))

        torch.save(model_state, os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.norm_type)))
            
        
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            best_accu_epoch = epoch

            model_state = {
                'net': net.state_dict(),
                'accu': val_accuracy,
                'epoch': epoch,
                'best_accu': best_accuracy,
                'best_accu_epoch': best_accu_epoch
            }
            
            print("Best Validation Accuracy Updated = {}%, Last Best = {}%".format(val_accuracy, best_accuracy))
            print("Saving Best Model Checkpoint:", now.strftime("%d-%m-%y %H:%M"))

            torch.save(model_state, os.path.join(args.checkpoint_dir, "best_val_checkpoint_{}.pth".format(args.norm_type)))
            


        with open(os.path.join(args.result_dir, "loss_tracker_{}_{}.json".format(args.norm_type, date_time)), "w") as outfile:
            json.dump(loss_tracker, outfile)

        with open(os.path.join(args.result_dir, "accuracy_tracker_{}_{}.json".format(args.norm_type, date_time)), "w") as outfile:
            json.dump(accuracy_tracker, outfile)
    

    return test_loader, net


def test(args, test_loader):
    net = load_checkpoint(args.checkpoint_dir, "best", args.norm_type)
    net = net.to(device)
    net.eval()    
    actuals = []
    predictions = []

    for idx, batch in enumerate(test_loader):
        
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        actuals.extend(labels.squeeze().tolist())
        outputs = net(images)
        predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())

    conf_mat = confusion_matrix(actuals, predictions)
    accu = accuracy_score(actuals, predictions)
    micro_f1 = round(f1_score(actuals, predictions, average='micro'), 4)
    macro_f1 = round(f1_score(actuals, predictions, average='macro'), 4)

    test_accuracy = "Test Accuracy = " + str(round(correct_pred/total_samples*100, 2)) + "%"
    micro_f1 = "Micro-F1 Score = " + str(micro_f1)
    macro_f1 = "Micro-F1 Score = " + str(macro_f1)

    test_result = [EXP_NAME, test_accuracy, micro_f1, macro_f1]
    with open(os.path.join(args.result_dir, "test_perormance.txt"), "w") as res:
        for r in test_result:
            res.writelines(r)
            res.writelines("\n")

    print("\n\n###########-------------------- Test Accuracy =", test_accuracy)



def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="ResNet for CIFAR-10 Classification.")

    # path to data files.
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory.")

    # path to result files.
    parser.add_argument("--result_dir", type=str, default="./results", help="Path to dataset directory.")

    # path to model checkpoints.
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to model checkpoints.")

    # batch size training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to be used during training.")

    # number of workers for dataloader
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers used for dataloading.")

    # max number of epochs
    parser.add_argument("--epochs", type=int, default=100, help="Number of workers used for dataloading.")

    # n
    parser.add_argument("--n", type=int, default=2, help="Number of residual blocks.")

    # r
    parser.add_argument("--r", type=int, default=10, help="Number of classes.")

    # normalization type
    parser.add_argument("--norm_type", type=str, default="torch_bn", help="Type of layer normalization to be used.")

    # data augmentation
    parser.add_argument("--aug", type=bool, default=True, help="Whether to perform data augmentation during training.")


    return parser



if __name__ == "__main__":
    
    #generate parser
    parser = get_parser()
    args = parser.parse_args()
    
    if args.aug:
        dir_name = args.norm_type + "_" + "aug"
    else:
        dir_name = args.norm_type + "_" + "noaug"
    
    print(dir_name)
    args.data_dir = os.path.relpath(args.data_dir)
    args.result_dir = os.path.join(os.path.relpath(args.result_dir), dir_name)
    args.checkpoint_dir = os.path.join(os.path.relpath(args.checkpoint_dir), dir_name)

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    EXP_NAME = args.norm_type + '_' + date_time  

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    assert os.path.isdir(args.data_dir), f"invalid directory: {args.data_dir}"
    print("\n", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device =", device)

    test_loader = train(args)
    test(args, test_loader, net)


