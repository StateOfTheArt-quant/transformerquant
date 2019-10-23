#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformerquant.modules.residual_bert import ResidualBERT

import torch
data = torch.randn(2,30,120)
model = ResidualBERT(d_model=120)
output = model(data)