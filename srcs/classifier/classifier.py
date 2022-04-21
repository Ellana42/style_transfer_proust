from transformers import CamembertModel, CamembertTokenizer
import torch
import numpy as np
import torch.nn as nn
import random
import time
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path


class CamemBertClassifier(nn.Module):

    def __init__(self, weights=None):
        super(CamemBertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base",
                                                            do_lower_case=True)
        self.bert = CamembertModel.from_pretrained("camembert-base")
        self.classifier = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(),
                                        nn.Linear(H, D_out))
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits
