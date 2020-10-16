from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import torch 
import json

class TextDataset(torch.utils.data.Dataset):
    """
    Creates a generic Pytorch Text Dataset.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_imdb_split(split_dir):
    """
    Reads IMDb data.

    params:
        - split_dir (str) : path to imdb data (train/test).
    returns:
        - text (list)   : list of input sentences.
        - labels (list) : list of labels 0 (neg) and 1 (pos).
    """

    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

def extract_simplified(source, target, lines=8021122):
    with open(target, 'w') as sr:
        cnt = 0;
        with open(source) as file:
            line = file.readline()
            while line:
                cnt += 1
                if cnt == lines:
                    break
                if cnt % 100_000 == 0:
                    print(f'Processed {cnt}')
                try:
                    line = file.readline().strip()
                    review_dict = json.loads(line)
                    simplified = {'text':  review_dict['text'], 'stars': review_dict['stars']}
                    simplified_text = json.dumps(simplified)
                    sr.write(f'{simplified_text}\n')
                except:
                    print(f'Could not parse {line}')

def read_yelp(data_path):
    """
    Reads Yelp data.
    Reads IMDb data.

    params:
        - data_path (str) : path to yelp data.
    returns:
        - train_text (list)   : list of input sentences training.
        - train_labels (list) : list of labels ranging from 0-4 (star rating).
        - test_text (list)   : list of input sentences for testing.
        - test_labels (list) : list of labels ranging from 0-4 (star rating).
    """

    #preprocess data
    extract_simplified(data_path, "./processed_yelp_reviews.json")

    # parse json into a dataframe
    yelp_df = pd.read_json("./processed_yelp_reviews.json", lines=True)
    # remove columns
    yelp_df = yelp_df[['text', 'stars']]
    texts = []
    labels = []
    for row in yelp_df.iterrows():
        print(row)
        texts.append(row['text'])
        # start labels from 0 (auto one hot encode)
        labels.append(int(row['stars']) -1)
    
    # create test data
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.1)

    return train_texts, test_texts, train_labels, test_labels



def load_and_preprocess(args, val=True):

    if args.dataset == "IMDb":
        train_texts, train_labels = read_imdb_split(args.train_path)
        test_texts, test_labels = read_imdb_split(args.test_path)
    elif args.dataset == "Yelp":
        train_texts, test_texts, train_labels, test_labels = read_yelp(args.data_path)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)


    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}