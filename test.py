import torch
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
    args = parser.parse_args()

    # Add gpu.
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # run baselines.
    args.train_path = "../imdb/aclImdb/train"
    args.test_path = "../imdb/aclImdb/test"
    args.dataset = "IMDb"
    # load dataset
    print("Loading Datasets")
    # load and preprocess the datasets.
    args.dataloaders = load_and_preprocess(args, test=True)

    print("Creating Model")
    # load pretrained BERT and push to GPU.
    #args.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    args.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="/work/cse896/atendle/imdb-train-base/")
    args.model.to(args.device)

    # load model 
    test(args)
