from pathlib import Path
from datasets import IMDbDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


def load_and_preprocess(train_path, test_path, val=True):

    train_texts, train_labels = read_imdb_split()
    test_texts, test_labels = read_imdb_split()

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}