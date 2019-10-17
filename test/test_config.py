#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from transformerquant.configs.base import PretrainedConfig
from transformerquant.configs.bert_config import BertConfig

to_save_path = os.path.join(os.path.expanduser('~'), "tmp")
if not os.path.isdir(to_save_path):
    os.mkdir(to_save_path)

# ====================================== #
# test config save_pretrained            #
# ====================================== #

config = BertConfig()
assert isinstance(config, PretrainedConfig) 
config.save_pretrained(to_save_path)

# ===================================== #
# test config from_pretrained           #
# ===================================== #
new_config = BertConfig.from_pretrained(to_save_path)

