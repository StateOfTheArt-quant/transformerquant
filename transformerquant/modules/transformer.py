#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from .attention.multi_head import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward, clones, LayerNorm


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=d_model, nhead=nhead)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=dim_feedforward, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.size = d_model
    
    def forward(self,x,mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        self.layers = clones(encoder_layer, num_layers)
        self.norm = LayerNorm(encoder_layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)