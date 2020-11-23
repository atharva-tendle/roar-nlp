import torch
import random
import numpy as np
import argparse
from transformers import BertForSequenceClassification
from utils import load_and_preprocess

def test(args):
    
    test_loss = []
    test_batches = 0

    args.model.eval()

    with torch.no_grad():
        for batch in args.dataloaders['test']:

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)

            outputs = args.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            test_loss.append(loss)
            test_batches += 1

    print("Testing loss: {}".format(sum(test_loss)/test_batches))
    
       
    

if __name__ == "__main__":
    # Parse Arguments.
    parser = argparse.ArgumentParser()
    parse.add_argument('--model-type', type=str, default="baseline", help="FIE")
    parser.add_argument('--model', type=str, default="/work/cse896/atendle/model-files/baseline", help='model path')
    parser.add_argument('--tokenizer', type=str, default="/work/cse896/atendle/model-files/baseline-tok", help='tokenizer')
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

    # run baselines.
    args.train_path = "../imdb/aclImdb/train"
    args.test_path = "../imdb/aclImdb/test"
    args.dataset = "IMDb"
    # load dataset
    print("Loading Datasets")

    if args.model_type == "baseline":
        # load and preprocess the datasets.
        args.dataloaders = load_and_preprocess(args, test=True)
    else:
        pass

    print("Creating Model")
    # load pretrained BERT and push to GPU.
    #args.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    args.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model)
    args.model.to(args.device)

    # load model 
    test(args)
