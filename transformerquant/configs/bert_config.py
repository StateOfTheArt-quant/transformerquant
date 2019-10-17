#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import sys
from io import open
from transformerquant.configs.base import PretrainedConfig

logger = logging.getLogger(__name__)

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "xx.json",
    'bert-large-uncased': "xx.json",}

class BertConfig(PretrainedConfig):
    
    @classmethod
    def create_from_json(cls, config_json_file):
        instance = cls()
        with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
        
        for key, value in json_config.items():
                instance.__dict__[key] = value
        return instance
    
    
    def __init__(self,
                 d_model=50,
                 num_layers=12,
                 nhead=12,
                 activation="gelu",
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 num_labels = 1,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers= num_layers
        self.nhead = nhead
        self.activation =activation
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels