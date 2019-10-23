#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from transformerquant.configs.bert_config import BertConfig
from transformerquant.models.residual_bert import BertForPreTraining, BertForSequenceClassification

to_save_path = os.path.join(os.path.expanduser('~'), "tmp")
if not os.path.isdir(to_save_path):
    os.mkdir(to_save_path)
    

bert_config = BertConfig()

bert_pretraining_model = BertForPreTraining(bert_config)

bert_sequence_classifier_model = BertForSequenceClassification(bert_config)