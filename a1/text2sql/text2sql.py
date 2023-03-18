import argparse
import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import json 
from time import time
from tqdm import tqdm
from datetime import datetime

from utils import *


class Text2SQL():

    def __init__(self, args, model_id):
        super(Text2SQL, self).__init__()
        self.model_id = model_id
        self.args = args
        self.embedding_type = args.embedding_type
        self.encoder_type = args.encoder_type
        self.embed_dim = args.embed_dim
        self.encoder_hidden_units = args.encoder_hidden_units
        self.decoder_hidden_units = args.decoder_hidden_units
        self.vocab = dict()
        self.encoder = None
        self.decoder = None
        self.embeddings = None

        pass

    def get_vocab(self):

        pass

    def get_encoder(self):

        pass

    def get_decoder(self):

        pass

    def beam_search(self):

        pass





