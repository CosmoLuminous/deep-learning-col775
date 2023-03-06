import argparse
import os
import numpy as np
import torch

def load_checkpoint(chkpt_dir, model_type, norm_type):
    if model_type == "best":
        model_name = os.path.join(chkpt_dir, "best_val_checkpoint_{}.pth".format(norm_type))
    else:
        model_name = os.path.join(chkpt_dir, "latest_checkpoint_{}.pth".format(norm_type))

    assert os.path.isfile(model_name), f"Model path/name invalid: {model_name}"

    net = torch.load(model_name)
    return net