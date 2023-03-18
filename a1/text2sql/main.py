import argparse
import os

def get_parser():
    """
    Generate a parameter parser
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Text2SQL.")

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
    parser.add_argument("--model_dim", type=int, default=768, help="Model dimensions.")

    # r
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units while using LSTM.")

    return parser



if __name__ == "__main__":
    
    #generate parser
    parser = get_parser()
    args = parser.parse_args()

    args.data_dir = os.path.relpath(args.data_dir)
    print(args)