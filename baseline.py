import torch
import argparse
from transformers import BertForSequenceClassification, AdamW



def imdb_baseline(args):

    print("Loading Datasets")
    args.dataloaders = load_and_preprocess(args.train_path, args.test_path, val=True)
    
    print("Creating Model")
    args.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    args.model.to(args.device)

    args.optim = AdamW(model.parameters(), lr=5e-5)
    
    print("Starting Training")
    train_and_validate(args)

def yelp_baseline(args):
    pass


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--dataset', type=str, default='IMDb', help='name of the dataset used for fine-tuning.')
    args = parser.parse_args()

    # Add gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # run baselines
    if args.dataset == "IMDb":

        args.train_path = "imdb/aclImdb/train"
        args.test_path = "imdb/aclImdb/test"
        imdb_baseline(args)

    elif args.dataset == "Yelp":

        yelp_baseline(args)