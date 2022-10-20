import sys, re
import numpy as np
import math

###############################################################################

def preprocess(s):
    """Tokenise a line"""
    o = re.sub('([^a-zA-Z0-9\']+)', ' \g<1> ', s.strip())
    return ['<BOS>'] + re.sub('  *', ' ', o).strip().split(' ')

###############################################################################

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

EMBEDDING_DIM = 4
CONTEXT_SIZE = 2 #!!!#
HIDDEN_DIM = 6

# Bigram Neural Network Model
class TrigramNNmodel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(TrigramNNmodel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size, bias = False)

    def forward(self, inputs):
        # compute x': concatenation of x1 and x2 embeddings
        embeds = self.embeddings(inputs).view(
                (-1,self.context_size * self.embedding_dim))
        # compute h: tanh(W_1.x' + b)
        out = torch.tanh(self.linear1(embeds))
        # compute W_2.h
        out = self.linear2(out)
        # compute y: log_softmax(W_2.h)
        log_probs = F.log_softmax(out, dim=1)
        # return log probabilities
        # BATCH_SIZE x len(vocab)
        return log_probs