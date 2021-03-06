import torch
import random
import numpy as np
import argparse
from transformers import BertForSequenceClassification
from utils import load_and_preprocess, load_and_preprocess_random

def test(args): 
    """
    Evaluates a pretrained model.

    params:
        args (argparser) : List of arguments for testing 

    """
    
    # tracking metrics
    test_loss = []
    test_batches = 0

    # set to eval mode.
    args.model.eval()
    with torch.no_grad():
        for batch in args.dataloaders['test']:
            
            # inputs for model
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            # get predictions
            outputs = args.model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # compute loss
            loss = outputs[0]
            
            test_loss.append(loss)
            test_batches += 1

    print("Testing loss: {}".format(sum(test_loss)/test_batches))
    
       
    

if __name__ == "__main__":
    # Parse Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default="baseline", help="FIE")
    parser.add_argument('--model', type=str, default="/work/cse896/atendle/model-files/baseline", help='model path')
    parser.add_argument('--tokenizer', type=str, default="/work/cse896/atendle/model-files/baseline-tok", help='tokenizer')
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

    # run baselines.
    args.train_path = "../imdb/aclImdb/train"
    args.test_path = "../imdb/aclImdb/test"
    args.dataset = "IMDb"
    # load dataset
    print("Loading Datasets")

    if args.model_type == "baseline":
        # load and preprocess the datasets.
        args.dataloaders = load_and_preprocess(args, test=True)
    elif args.model_type == "random":
        # load and preprocess the datasets.
        args.dataloaders = load_and_preprocess(args, test=True)

    print("Creating Model")
    # load pretrained BERT and push to GPU.
    #args.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    args.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model)
    args.model.to(args.device)

    # load model 
    test(args)
