#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding

class TransformerEmbedding(nn.Module):
    
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
    
    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)