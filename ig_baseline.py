import torch
import random
import numpy as np
import argparse
from transformers import BertForSequenceClassification, AdamW
from utils import load_and_preprocess_ig
from train import train_and_validate
from test import test



def ig_baseline(args):
    """
    Creates a baseline training run for the IMDb/Yelp dataset with masks generated by Integrated Gradients.

    params:
        args (argparser) : List of arguments for training 

    """

    print("Loading Datasets")
    # load and preprocess the datasets.
    args.dataloaders = load_and_preprocess_ig(args)
    
    print("Creating Model")
    # load pretrained BERT and push to GPU.
    args.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    args.model.to(args.device)

    # create optimizer
    args.optim = AdamW(args.model.parameters(), lr=5e-5)
    
    print("Starting Training")
    # run training.
    
    if args.label_dependent_masking:
        save_path = "/work/vinod/gwirka/classes/nlp/roar-cache/imdb-train-ig_{}_ldm".format(int(args.t * 10))
    else:
        save_path = "/work/vinod/gwirka/classes/nlp/roar-cache/imdb-train-ig_{}".format(int(args.t * 10))
    
    args = train_and_validate(args, save_path=save_path)



if __name__ == "__main__":
    # Parse Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--dataset', type=str, default='IMDb', help='name of the dataset used for fine-tuning.')
    parser.add_argument('--t', type=float, default=0.1, help='proportion of each input to mask with [UNK] tokens.')
    parser.add_argument('--label_dependent_masking', action='store_true', help='mask based on instance label, instead of absolute value of attributions.')
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

    # run ig baselines.
    if args.dataset == "IMDb":
        # path to IMDb data.
        args.train_path = "/work/vinod/gwirka/classes/nlp/roar-nlp/data/imdb/aclImdb/train"
        args.test_path = "/work/vinod/gwirka/classes/nlp/roar-nlp/data/imdb/aclImdb/train"
        ig_baseline(args)

    elif args.dataset == "Yelp":
        # path to Yelp data.
        args.data_path = "../yelp/yelp_academic_dataset_review.json"
        ig_baseline(args)