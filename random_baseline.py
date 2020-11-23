import torch
import random
import numpy as np
import argparse
from transformers import BertForSequenceClassification, AdamW
from utils import load_and_preprocess_random
from train import train_and_validate
from test import test



def random_baseline(args):
    """
    Creates a baseline training run for the IMDb/Yelp dataset.

    params:
        args (argparser) : List of arguments for training 

    """

    print("Loading Datasets")
    # load and preprocess the datasets.
    args.dataloaders = load_and_preprocess_random(args, t=0.8)
    
    print("Creating Model")
    # load pretrained BERT and push to GPU.
    args.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    args.model.to(args.device)

    # create optimizer
    args.optim = AdamW(args.model.parameters(), lr=5e-5)
    
    print("Starting Training")
    # run training.
    args = train_and_validate(args, save_path="/work/cse896/atendle/imdb-train-random_chancee_80")



if __name__ == "__main__":
    # Parse Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--dataset', type=str, default='IMDb', help='name of the dataset used for fine-tuning.')
    args = parser.parse_args()

    # Add gpu.
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set seed for reproducibility.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # gpu training specific seed settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run random baselines.
    if args.dataset == "IMDb":
        # path to IMDb data.
        args.train_path = "../imdb/aclImdb/train"
        args.test_path = "../imdb/aclImdb/test"
        random_baseline(args)

    elif args.dataset == "Yelp":
        # path to Yelp data.
        args.data_path = "../yelp/yelp_academic_dataset_review.json"
        random_baseline(args)
