#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .embedding.position import ResidualPositionalEmbedding


class ResidualBERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    
    def __init__(self, d_model, num_layders=12, nhead=12, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layders
        self.nhead = nhead
        self.dim_feedforward = d_model * 4
        
        self.embedding = ResidualPositionalEmbedding(d_model=d_model)
        transformer_encoder_layder = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=self.dim_feedforward, dropout=dropout, activation=activation)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layder, num_layders = num_layders)
    
    def forward(self, x, mask=None):
        
        x = self.embedding(x)
        for transformer in self.transformer_encoder:
            x = transformer.forward(x, mask)
        return x
        