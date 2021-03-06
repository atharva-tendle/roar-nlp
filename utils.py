from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import torch 
import pickle

# STOP WORDS for TF-IDF masking
STOP_WORDS = {'haven', 'if', 'has', 'is', 'are', 've', 'did', 'then', 'and', 'o', "won't", 'shan', 'until', "couldn't", 'most', "haven't", 'very', 'not', "you're", 'from', 'wouldn', "you'd", 'y', "weren't", 'the', 'you', 'needn', 'too', 'because', 'any', 'just', 'mustn', 'doing', 'or', 'him', 'her', 'wasn', 'by', 'in', 'theirs', "should've", 'some', 'now', 'ain', 'above', 'both', 'don', "don't", 's', 'ours', 'once', 'they', 'am', 'there', 'so', 'weren', 'himself', 'she', "needn't", 'shouldn', 'i', 'herself', "it's", 'when', 'other', 'can', 'didn', "hadn't", 'no', 'over', 'few', 'down', 'here', "mustn't", 'them', 'under', 'that', 'be', 'your', 'where', 'aren', "you'll", 'below', 'into', 'ourselves', "aren't", 'doesn', 'themselves', 'my', 'its', 'who', 'as', "hasn't", 'further', 'our', 'own', 'it', 'being', 'on', "you've", 'of', 'such', 'those', 'all', 'yourselves', 'should', 'while', 'were', 'been', "doesn't", 'does', 'out', 'what', 'during', 'his', 'he', 'had', 'through', 'an', 'their', 'again', "she's", 'after', 'this', 'these', 'but', 'we', 'me', 'how', 'will', "mightn't", 'yours', 'itself', 'against', 'ma', 'do', 'having', 'nor', 'm', 'hadn', "wasn't", 'before', 'between', 'a', 'won', "didn't", 'myself', 'more', 't', 're', 'd', "wouldn't", "shan't", 'each', 'isn', 'for', "isn't", 'll', "shouldn't", "that'll", 'with', 'yourself', 'to', 'couldn', 'at', 'mightn', 'whom', 'which', 'why', 'same', 'up', 'only', 'than', 'have', 'about', 'off', 'hers', 'hasn', 'was'}


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

def read_imdb_split(split_dir, return_filenames=False):
    """
    Reads IMDb data.

    params:
        - split_dir (str) : path to imdb data (train/test).
    returns:
        - text (list)   : list of input sentences.
        - labels (list) : list of labels 0 (neg) and 1 (pos).
    """
    # get directory
    split_dir = Path(split_dir)
    texts = []
    labels = []
    filenames = []
    # load imdb data by labels
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)
            filenames.append(text_file.parts[-1])
    if return_filenames:
        return texts, labels, filenames
    else:
        return texts, labels

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

    # parse json into a dataframe
    yelp_df = pd.read_json(data_path, lines=True)
    # remove columns
    yelp_df = yelp_df[['text', 'stars']]

    texts, labels = [], []

    for idx, row in yelp_df.iterrows():
        texts.append(row['text'])
        # start labels from 0 (auto one hot encode)
        labels.append(int(row['stars']) -1)
    
    # create test data
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.1)

    return train_texts, test_texts, train_labels, test_labels


def load_and_preprocess(args, test=False):
    """
    Loads Data for baseline experiment.
    """
    # load dataset.
    if args.dataset == "IMDb":
        train_texts, train_labels = read_imdb_split(args.train_path)
        test_texts, test_labels = read_imdb_split(args.test_path)
    elif args.dataset == "Yelp":
        train_texts, test_texts, train_labels, test_labels = read_yelp(args.data_path)

    # load tokenizer.
    if test:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained(pretrained_model_name_or_path=args.tokenizer)
    else:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained('bert-base-uncased')
    
    # create encodings.
    train_encodings = tokenizer(train_texts, truncation=True, max_length=128, padding='max_length')
    #val_encodings = tokenizer(val_texts, truncation=True, max_length=128, padding='max_length')
    test_encodings = tokenizer(test_texts, truncation=True, max_length=128, padding='max_length')

    if test:
        pass
    else:
        tokenizer.save_pretrained("/work/cse896/atendle/imdb-train-base-tok")

    # creat torch datasets.
    train_dataset = TextDataset(train_encodings, train_labels)
    #val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)


    # create dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return {'train': train_loader, 'test': test_loader}

def load_and_preprocess_random(args, test=False, t=0.1):
    """
    Loads Data for Random FIE experiments
    """

    # load dataset.
    if args.dataset == "IMDb":
        train_texts, train_labels = read_imdb_split(args.train_path)
        test_texts, test_labels = read_imdb_split(args.test_path)
    elif args.dataset == "Yelp":
        train_texts, test_texts, train_labels, test_labels = read_yelp(args.data_path)

    # create validation split.
    #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    train_texts = random_text_process(train_texts, t=t)
    test_texts = random_text_process(test_texts, t=t)

    # load tokenizer.
    if test:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained(pretrained_model_name_or_path=args.tokenizer)
    else:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained('bert-base-uncased')
    
    # create encodings.
    train_encodings = tokenizer(train_texts, truncation=True, max_length=128, padding='max_length')
    #val_encodings = tokenizer(val_texts, truncation=True, max_length=128, padding='max_length')
    test_encodings = tokenizer(test_texts, truncation=True, max_length=128, padding='max_length')

    if test:
        pass
    else:
        tokenizer.save_pretrained("/work/cse896/atendle/imdb-train-random_chance_80-tok")

    # creat torch datasets.
    train_dataset = TextDataset(train_encodings, train_labels)
    #val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)


    # create dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return {'train': train_loader, 'test': test_loader}

def load_and_preprocess_tfidf(args, test=False, t=0.1):
    """
    Loads Data for TF-IDF FIE experiments.
    """

    # load dataset.
    if args.dataset == "IMDb":
        train_texts, train_labels = read_imdb_split(args.train_path)
        test_texts, test_labels = read_imdb_split(args.test_path)
    elif args.dataset == "Yelp":
        train_texts, test_texts, train_labels, test_labels = read_yelp(args.data_path)

    # create validation split.
    #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    train_texts = tfidf_mask(train_texts, t=t)
    test_texts = tfidf_mask(test_texts, t=t)

    # load tokenizer.
    if test:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained(pretrained_model_name_or_path='/work/cse896/atendle/imdb-train-tfidf_60-tok')
    else:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained('bert-base-uncased')
    
    # create encodings.
    train_encodings = tokenizer(train_texts, truncation=True, max_length=128, padding='max_length')
    #val_encodings = tokenizer(val_texts, truncation=True, max_length=128, padding='max_length')
    test_encodings = tokenizer(test_texts, truncation=True, max_length=128, padding='max_length')

    if test:
        pass
    else:
        tokenizer.save_pretrained("/work/cse896/atendle/imdb-train-tfidf_60-tok")

    # creat torch datasets.
    train_dataset = TextDataset(train_encodings, train_labels)
    #val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)


    # create dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return {'train': train_loader, 'test': test_loader}


def load_and_preprocess_ig(args, test=False, t=0.1, label_dependent_masking=False):
    # load dataset.
    
    if args.t:
        t=args.t
    if args.label_dependent_masking:
        label_dependent_masking=args.label_dependent_masking
    
    print("t: {}".format(t))
    print("Using ldm? {}".format(label_dependent_masking))
    
    print("Loading data.")
    if args.dataset == "IMDb":
        train_texts, train_labels, train_filenames = read_imdb_split(args.train_path, return_filenames=True)
        test_texts, test_labels = read_imdb_split(args.test_path)
    elif args.dataset == "Yelp":
        train_texts, test_texts, train_labels, test_labels = read_yelp(args.data_path)
        
    # create validation split.
    #train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)


    # load tokenizer.
    print("Loading tokenizer.")
    if test:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained(pretrained_model_name_or_path='/work/vinod/gwirka/classes/nlp/roar-cache/model-files/baseline/imdb-train-base-tok')
    else:
        tokenizer = BertTokenizerFast(vocab_file="./bert-base-uncased.txt").from_pretrained('bert-base-uncased')
    
    # create encodings.
    print("Creating encodings.")
    train_encodings = tokenizer(train_texts, truncation=True, max_length=128, padding='max_length')
    #val_encodings = tokenizer(val_texts, truncation=True, max_length=128, padding='max_length')
    test_encodings = tokenizer(test_texts, truncation=True, max_length=128, padding='max_length')

    # load attributions
    print("Applying masks.")
    infile = open("/work/vinod/gwirka/classes/nlp/roar-nlp/data/imdb/aclImdb/train/imdb_train_gpu_attr_dict.p", "rb")
    attributions = pickle.load(infile)
    infile.close()
    
    # apply masks
    
    print("n : {}".format(len(train_encodings['input_ids'])))
    
    for i in range(len(train_encodings['input_ids'])):
        if i%1000 == 0: print("~{:.2f}%".format(i/len(train_encodings['input_ids'])))
        num_to_mask = int(len(train_encodings['input_ids'][i]) * t)
        
        # if i == 0:
        #     print("Num masked on one example: {}".format(num_to_mask))
        #     print(train_encodings['input_ids'][0])
        
        # print(train_encodings['input_ids'][0])
        # print(tokenizer.convert_ids_to_tokens(train_encodings['input_ids'][i]))
        
        attribution = attributions[train_filenames[i]][:-3] + [0.0] 
        """
        The attributions are of length 130. The first and last elements need to remain, because the tokenizer does
        generate [CLS] and [SEP] tokens (which have attribution weights of 0.0), so the last two non-special tokens
        need to be discarded.
        """
        
        if label_dependent_masking:
            """
            If an instance is labeled positive, remove the tokens with the most positive attributions, 
            if negative remove the most negative.
            
            This approach is more adversarial, and should push the model towards making incorrect choices.
            """
            
            sorted_attribution_idxes = np.argsort(attribution) #smallest to largest
            
            if(train_labels[i]):
                for j in np.flip(sorted_attribution_idxes, axis=0)[:num_to_mask]:
                    train_encodings['input_ids'][i][j] = 100 # replace with unk token id
            else:
                for j in sorted_attribution_idxes[:num_to_mask]:
                    train_encodings['input_ids'][i][j] = 100 # replace with unk token id
        else:
            """
            Replace the tokens whose attributions have the highest magnitude, regardless of label.
            
            This approach removes meaningful information and pushes the model towards random guessing.
            """
            
            sorted_attribution_idxes = np.argsort([ abs(x) for x in attribution ])
            
            
            for j in np.flip(sorted_attribution_idxes, axis=0)[:num_to_mask]:
                train_encodings['input_ids'][i][j] = 100 # replace with unk token id
                
        # print(tokenizer.convert_ids_to_tokens(train_encodings['input_ids'][i]))

    if test:
        pass
    else:
        tokenizer.save_pretrained("/work/vinod/gwirka/classes/nlp/roar-cache/model-files/ig/imdb-train-ig_10-tok")

    # creat torch datasets.
    train_dataset = TextDataset(train_encodings, train_labels)
    #val_dataset = TextDataset(val_encodings, val_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    # create dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return {'train': train_loader, 'test': test_loader}

def random_text_process(train_texts, t=0.1):
    """
    Masks inputs randomly
    """
    new_train_texts  = []

    for i in range(len(train_texts)):
        # get the current text
        current_text = train_texts[i].split(" ")
        # number of words to mask
        n = int(len(current_text) * t)
        # indices of words to mask
        indices = np.random.choice(len(current_text), n, replace=False)

        # mask words
        for index in indices:
            current_text[index] = "[UNK]"

        new_train_texts.append(' '.join(current_text))

    return new_train_texts

def get_tfidf_rankings(feature_array, tfidf_matrix):
    """
    Generates TF-IDF rankings for given input.
    """
    tfidf_rankings = []

    for index in tfidf_matrix.indices:
        if feature_array[index] not in STOP_WORDS:
            tfidf_rankings.append((feature_array[index], tfidf_matrix[0, index]))

    return sorted(tfidf_rankings, key=lambda x:x[1])

def tfidf_mask(texts, t=0.5):
    """
    Masks inputs using TF-IDF rankings.
    """

    # create TF-IDF vectorizer.
    vectorizer = TfidfVectorizer()
    # fit the corpus.
    vectorizer.fit(texts)
    # get list of feature names.
    feature_array = np.array(vectorizer.get_feature_names())
    # list to store masked inputs.
    new_texts  = []
    
    for idx, train_text in enumerate(texts):
        current_text = train_text.split(" ")
        matching_text = train_text.lower().split(" ")
        tfidf_matrix = vectorizer.transform(matching_text)
        num_maskings = int(len(matching_text) * t)
        masked = 0
        tfidf_rankings = get_tfidf_rankings(feature_array, tfidf_matrix)

        for tfidf_w in tfidf_rankings:
            # perfect match.
            try:
                w_index = matching_text.index(tfidf_w[0])
                current_text[w_index] = "[UNK]"
                masked += 1
                if masked == num_maskings:
                    break
            # match with punctuations.
            except:
                for idx, word in enumerate(matching_text):
                    if tfidf_w[0] in word:
                        current_text[idx] = "[UNK]"
                        masked += 1
                if masked == num_maskings:
                    break
                
                
        new_texts.append(' '.join(current_text))

    return new_texts





