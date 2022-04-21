import glob
import json
import re
import pickle
import os
import shutil
import functools
import torch

import numpy as np

from nltk.tokenize import RegexpTokenizer
from torch import nn

from torch.utils.data import Dataset, DataLoader


tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')


class TextDataset(Dataset):
    '''Class to be used to generate a pytorch dataloader of sentences extracted from french articles '''

    def __init__(self, paths, tokenizer_function, vocab_stoi, sequence_size=10, pad_token="<pad>", start_token="<s>", end_token="</s>"):
        '''Instantiate french article dataset class
        path_to_articles : list containing path
        tokenizer_function : callable function which tokenizes a string
        vocab_stoi : stoi indexing
        sequence_size : len of sentence sampled without start of sentence and end of sentence token
        pad_token : padding token default <pad>
        start_token : start of sentence token default <s>
        end_token : end of sentence token default </s>'''

        super().__init__()

        self.paths = paths  # list containing list of paths to articles and proust
        self.len = len(paths)  # dataset len
        self.tokenizer_function = tokenizer_function  # tokenizer used
        self.sequence_size = sequence_size
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.stoi = vocab_stoi

    def __len__(self):
        '''number of elements in the dataset'''
        return(self.len)

    def pad(self, token_sequence):
        '''Function to pad sequence
        token_sequence : list of tokens of len < self.sequence_size'''

        pad_size = self.sequence_size-len(token_sequence)
        result = [self.start_token] + token_sequence + \
            [self.pad_token for i in range(pad_size)] + [self.end_token]

        return result

    def sample(self, tokens):
        ''' sample sequence from token sequence
        tokens : list of tokens'''

        nb_tokens = len(tokens)
        starting_index = np.random.randint(nb_tokens)

        if starting_index + self.sequence_size < nb_tokens:
            result = [self.start_token] + tokens[starting_index: starting_index +
                                                 self.sequence_size] + [self.end_token]

        else:
            result = tokens[starting_index: nb_tokens]
            result = self.pad(result)

        return result

    def __getitem__(self, idx):
        '''get a sentence of size sequence_size from file of index idx
        '''

        text_file = read_json(self.paths[idx])
        label = text_file["label"]
        text = text_file["text"]

        tokens = self.tokenizer_function(text)

        sequence = [self.stoi[token] for token in self.sample(tokens)]

        return(sequence, label)

    @staticmethod
    def read_json(file_path):
        '''Read Json file from path
        file_path : string, path to json file'''
        with open(file_path, encoding='utf-8') as f:
            json_file = json.load(f)
        return(json_file)

    @staticmethod
    def tokenize_sentence(sentence: str, tokenizer: RegexpTokenizer):
        '''Simple tokenizer, removes or replaces special characters
        sentence : str sentence to be tokenized
        tokenizer : tokenizer with tokenize method '''

        # Lower capital leters
        tokenized = sentence.lower()
        # Change special character
        tokenized = re.sub("’", "'", tokenized)
        # Remove unwanted characters
        tokenized = re.sub(
            "(@\w*\\b\s?|#\w*\\b\s?|&\w*\\b\s?|\n\s?|\\\\|\<|\>|\||\*)", "", tokenized)
        tokenized = re.sub("\/", "", tokenized)
        # Replace articles since model does not embed contractions well
        tokenized = re.sub("l'", "le ", tokenized)
        tokenized = re.sub("d'", "de ", tokenized)
        tokenized = re.sub("j'", "je ", tokenized)
        tokenized = re.sub("qu'", "que ", tokenized)
        tokenized = re.sub("t'", "te ", tokenized)
        tokenized = re.sub("c'", "ce ", tokenized)
        # Tokenize sentence
        tokenized = tokenizer.tokenize(tokenized)
        return(tokenized)
