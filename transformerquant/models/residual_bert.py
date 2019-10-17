#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn

from transformerquant.models.base import PreTrainedModel
from transformerquant.modules.residual_bert import ResidualBERT
from transformerquant.modules.activation.activations import ACT2FN
from transformerquant.configs.bert_config import BertConfig

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'residual-bert': "",
    'residual-bert-large': ""
}

BertLayerNorm = torch.nn.LayerNorm

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

class BertPooler(nn.Module):
    def __init__(self, in_features, out_features):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # sample size [batch_size, sequence_len, dim] -> [batch_size, dim]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    
    def __init__(self, in_features, activation_func="relu", layer_norm_eps=1e-6):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(in_features,in_features)
        self.transform_act_fn = ACT2FN[activation_func]
        self.LayderNorm = BertLayerNorm(in_features, eps=layer_norm_eps)
    
    def forward(self, x):
        x = self.dense(x)
        x = self.transform_act_fn(x)
        x = self.LayerNorm(x)
        return x
        
class BertLMPredictionHead(nn.Module):
    def __init__(self,in_features, out_features, activation_func="relu", layer_norm_eps=1e-6):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(in_features=in_features, activation_func=activation_func, layer_norm_eps=layer_norm_eps)
        self.decoder = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
# ============================================= #
# bert model(bert encoder)                      #
# ============================================= #
class ResidualBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(ResidualBertModel, self).__init__(config)
        
        self.encoder = ResidualBERT(d_model = config.d_model,
                                    num_layers=config.num_layers,
                                    nhead=config.nhead,
                                    dropout=config.attention_probs_dropout_prob,
                                    activation=config.activation)
        self.pooler = BertPooler(in_features=config.d_model, out_features=config.d_model)
        self.init_weight()
    
    def forward(self, x, mask=None):
        x = self.encoder(x,mask)
        x = self.pooler(x)
        return x


# ========================================== #
# bert for pretraining                       #
# ========================================== #
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        
        self.bert = ResidualBertModel(config)
        self.cls = BertLMPredictionHead(in_features=config.d_model, out_features=config.num_labels, activation_func=config.activation, layer_norm_eps=config.layer_norm_eps)
        
        self.init_weight()
    
    def forward(self, x, mask=None):
        x = self.bert(x, mask)
        x = self.cls(x)
        return x
    
# ========================================== #
# bert for downsteam task                    #
# ========================================== #
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = ResidualBERT(d_model = config.d_model,
                                 num_layers=config.num_layers,
                                 nhead=config.nhead,
                                 dropout=config.attention_dropout_prob,
                                 activation=config.activation)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
    
    def forward(self, x, mask):
        x = self.bert(x, mask)
        logit = self.classifier(x)
        return logit 