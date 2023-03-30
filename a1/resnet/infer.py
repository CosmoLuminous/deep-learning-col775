import json
import argparse
import os
import numpy as np
from collections import Counter, defaultdict
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def load_checkpoint(file_path):

    assert os.path.isfile(file_path), f"Model path/name invalid: {model_name}"
    
    # model_dict = torch.load(model_name)
    # net = ResNet(norm_type = args.norm_type) 
    # net.load_state_dict(model_dict['net'])
    net = torch.load(file_path)
    print("\n|--------- Model Load Success.", file_path)

    return net

class CIFAR10(Dataset):
    def __init__(self, file_path, img_transform=None):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, header=None)

        self.num_samples = len(self.data)
        self.images = self.data.to_numpy()
        self.img_transform = img_transform

        print("Number of samples in test file =", self.num_samples)

    def __len__(self):        
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.reshape(3,32,32)
        image = image/255
        image = image.transpose(1,2,0)

        image = self.img_transform(image)

        return {'images': image}


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device =", device)

    norm1 = (0.4914, 0.4822, 0.4465)
    norm2 = (0.2023, 0.1994, 0.2010)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm1, norm2)])
    test_dataset = CIFAR10(args.test_data_file, img_transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers
    )

    model = load_checkpoint(args.model_file)
    model = model.to(device)

    predictions = []

    for idx, batch in enumerate(test_loader):

        images = batch['images']
        images = images.type(torch.FloatTensor).to(device)

        outputs = model(images)
        predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())

    df = pd.DataFrame(predictions)
    df.to_csv(args.output_file, index=False, header=False)

# class CIFAR10_self_check(Dataset):
#     def __init__(self, file_path, img_transform=None):
#         self.file_path = file_path
#         self.data = pd.read_csv(file_path, header=None)

#         self.num_samples = len(self.data)
#         self.images = self.data.iloc[:,1:].to_numpy()
#         self.labels = list(self.data.iloc[:,0])
#         self.img_transform = img_transform

#         print("Number of samples in test file =", self.num_samples)

#     def __len__(self):        
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         image = np.array(image).reshape(3,32,32)
#         image = image/255
#         image = image.transpose(1,2,0)
#         image = self.img_transform(image)
#         label = self.labels[idx]

#         return {'images': image, 'labels': label}


# def infer_self_check(args):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Running on device =", device)

    # norm1 = (0.4914, 0.4822, 0.4465)
    # norm2 = (0.2023, 0.1994, 0.2010)
    # test_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(norm1, norm2)])
    # test_dataset = CIFAR10_self_check(args.test_data_file, img_transform=test_transforms)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers
    # )

    # model = load_checkpoint(args.model_file)
    # model = model.to(device)
    # actuals = []
    # predictions = []

    # for idx, batch in enumerate(test_loader):

    #     images = batch['images']
    #     labels = batch['labels']
    #     images = images.type(torch.FloatTensor).to(device)
    #     # print(images.shape, type(images))
    #     outputs = model(images)
    #     predictions.extend(torch.argmax(outputs, dim=1).squeeze().tolist())
    #     actuals.extend(labels.squeeze().tolist())

    # df = pd.DataFrame(predictions)
    # df.to_csv(args.output_file, index=False, header=False)
    # accu = round(accuracy_score(actuals, predictions)*100,2)
    # micro_f1 = round(f1_score(actuals, predictions, average='micro')*100, 2)
    # macro_f1 = round(f1_score(actuals, predictions, average='macro')*100, 2)
    # test_accuracy = "Accuracy = " + str(accu) + "%"
    # test_micro_f1 = "Micro-F1 Score = " + str(micro_f1)
    # test_macro_f1 = "Macro-F1 Score = " + str(macro_f1)
    # test_result = [
    #     args.model_type, test_accuracy, test_micro_f1, test_macro_f1
    #  ]
    # print(test_result)
    # with open(args.model_type + "_result.txt", "w") as res:
    #     for r in test_result:
    #         res.writelines(r)
    #         res.writelines("\n")


def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Inference ResNet for CIFAR-10 Classification.")

    # path to data files.
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory.")
    
    # path to data files.
    parser.add_argument("--model_file", type=str, help="Path to trained model file.")

    # path to data files.
    parser.add_argument("--test_data_file", type=str, help="Path to a csv with each line representing an image.")

    # path to data files.
    parser.add_argument("--output_file", type=str, help="file containing the prediction in the same order as in the input csv.")


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

    # normalization type
    parser.add_argument("--normalization", type=str, default="torch_bn", help="Type of layer normalization to be used.")


    # data augmentation
    parser.add_argument("--aug", type=bool, default=True, help="Whether to perform data augmentation during training.")

    # data augmentation
    # parser.add_argument("--comparison_plots_only", type=bool, default=False, help="Whether to perform data augmentation during training.")


    return parser



if __name__ == "__main__":
    
    #generate parser
    parser = get_parser()
    args = parser.parse_args()

    if args.normalization == "inbuilt":
        args.model_type = "torch_bn"
    else:
        args.model_type = args.normalization
    
    if not args.aug:
        dir_name = args.norm_type + "_" + "noaug"
    else:
        dir_name = args.norm_type
    
    # print(dir_name)
    args.data_dir = os.path.relpath(args.data_dir)
    
    args.result_dir = os.path.join(os.path.relpath(args.result_dir), dir_name)
    args.checkpoint_dir = os.path.join(os.path.relpath(args.checkpoint_dir), dir_name)

    infer(args)
    # infer_self_check(args)