import pandas as pd
from pathlib import Path
import os
import re
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import string


class StyleDataset:

    def __init__(self,
                 tokenizer,
                 split_ratios=(0.80, 0.10),
                 batch_size=32,
                 max_len=175):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.data_dir = Path('datasets')
        self.path = self.data_dir / Path('dataset.csv')
        self.df = pd.read_csv(self.path, sep='|', index_col=0)
        self.format()
        self.dataset = self.tokenize()
        self.split = self.get_ratios()
        self.train_df, self.val_df, self.test_df = random_split(
            self.dataset, self.split)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_dataloaders(
        )

    def format(self):
        print("Formatting data ...")
        self.df = self.df[['label', 'text']]
        self.df.text = self.df.text.apply(str)
        self.df = self.df.rename(columns={'label': 'y', 'text': 'X'})
        self.df.X = self.df.X.apply(self.preprocessing)
        # self.df = self.df.sample(1500)

    def preprocessing(self, text):
        text = re.sub("’", "'", text)
        #Remove unwanted characters
        text = re.sub(
            "(@\w*\\b\s?|#\w*\\b\s?|&\w*\\b\s?|\n\s?|\\\\|\<|\>|\||\*)", "",
            text)
        text = re.sub("\/", "", text)
        #Replace articles since model does not embed contractions well
        text = re.sub("l'", "le ", text)
        text = re.sub("d'", "de ", text)
        text = re.sub("j'", "je ", text)
        text = re.sub("qu'", "que ", text)
        text = re.sub("t'", "te ", text)
        text = re.sub("c'", "ce ", text)
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(
            ' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return (text)

    def get_ratios(self):
        train_ratio, val_ratio = self.split_ratios
        len_df = len(self.dataset)
        train_size = int(len_df * train_ratio)
        val_size = int(len_df * val_ratio)
        test_size = len_df - train_size - val_size
        return train_size, val_size, test_size

    def get_dataloaders(self):
        train_dataloader = DataLoader(self.train_df,
                                      sampler=RandomSampler(self.train_df),
                                      batch_size=self.batch_size)
        validation_dataloader = DataLoader(self.val_df,
                                           sampler=SequentialSampler(
                                               self.val_df),
                                           batch_size=self.batch_size)
        test_dataloader = DataLoader(self.test_df,
                                     sampler=SequentialSampler(self.test_df),
                                     batch_size=self.batch_size)
        return train_dataloader, validation_dataloader, test_dataloader

    def __len__(self):
        return len(self.df)

    def tokenize(self):
        input_ids = []
        attention_masks = []

        for sentence in self.df.X.values:
            encoded_sent = self.tokenizer.encode_plus(
                text=sentence,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=self.max_len,  # Max length to truncate/pad
                truncation=True,
                padding='max_length',
                return_attention_mask=True  # Return attention mask
            )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        dataset = TensorDataset(input_ids, attention_masks,
                                torch.tensor(self.df.y.values))
        return dataset

    def get_max_len(self, tokenizer):
        max_len = 0
        for sentence in self.data.X:
            input_ids = tokenizer.encode(sentence, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        return (max_len)
