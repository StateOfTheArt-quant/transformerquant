#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# reference: https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py

class StructuredSelfAttention(torch.nn.Module):
    
    @staticmethod
    def create(config):
        input_dim = getattr(config, 'input_dim', None)
        lstm_hid_dim = getattr(config, 'lstm_hid_dim', None)
        output_dim = getattr(config, 'output_dim', None)
        d_a = getattr(config, 'd_a', 100)
        r = getattr(config, 'r', 5)
        n_layers = getattr(config, 'n_layers',1)
        bidirectional = getattr(config, 'bidirectional', False)
        dropout = getattr(config, 'dropout', False)
        return StructuredSelfAttention(input_dim, lstm_hid_dim, output_dim, d_a, r, n_layers, bidirectional, dropout)
        
    
    def __init__(self, input_dim, lstm_hid_dim, output_dim, d_a=100, r=5, n_layers=1, bidirectional=False, dropout=0):
        """
        Initialize parameters suggested in paper
        
        Args:
            input_dim    : {int} the vector dimension of each element in the input sequence
            lstm_hid_dim : {int} hidden dimension for lstm
            output_dim   : {int} number of class
            d_a          : {int} hidden dimension of the dense layer for calc attention
            r            : {int} attention-hops or attention heads
            #type         : [0,1] 0-->binary classification 1-->multiclass classification
        
        Return:
            output, attention
        """
        super(StructuredSelfAttention, self).__init__()
        self.lstm_hid_dim = lstm_hid_dim
        self.output_dim = output_dim
        self.r = r
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        
        #model
        #hint: the output dim of lstm with bidirection is: lstm_hid_dim x 2 
        self.lstm = nn.LSTM(input_dim, lstm_hid_dim, n_layers, batch_first=True, bidirectional=bidirectional) # pay attention the bidirection args
        self.linear_first = nn.Linear(lstm_hid_dim * self.n_directions, d_a) 
        self.linear_first.bias.data.fill_(0)
        self.linear_second = nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.linear_final = nn.Linear(lstm_hid_dim * self.n_directions, output_dim)
        self.dropout = nn.Dropout(p = dropout)
    
    def init_hidden(self, input):
        h0 = torch.zeros(self.n_layers* self.n_directions, input.size(0), self.lstm_hid_dim).to(input.device)
        c0 = torch.zeros(self.n_layers* self.n_directions, input.size(0), self.lstm_hid_dim).to(input.device)
        return (h0,c0)
    
    def forward(self, x):
        init_hidden = self.init_hidden(x)
        
        x = self.dropout(x)
            
        # batch_size x seq_window x (lstm_hid_dim * self.n_bidirections)
        lstm_outputs, _ =self.lstm(x, init_hidden)
        lstm_outputs = self.dropout(lstm_outputs)
            
        # ================================ #
        # calc the self-attention          #
        # ================================ #
        x = torch.tanh(self.linear_first(lstm_outputs))
        x = self.linear_second(x)
        x = F.softmax(x, 1)
        attention = x.transpose(1,2) #batch_size x r x seq_window
        
        sentence_embeddings =attention@lstm_outputs   #batch_size x r x lstm_hid_dim # == attention.bmm(lstm_outputs)  #batch matrix-matrix product of matrix
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r  # batch_size x lstm_hid_dim
        output = self.linear_final(avg_sentence_embeddings)

        return output, attention #batch_size x output_dim

if __name__ == "__main__":
    input_dim = 20
    lstm_hid_dim=50
    output_dim = 2
    batch_size = 4
    sequence_window = 5
    
    batch_x = torch.randn((batch_size, sequence_window, input_dim))
    
    model = StructuredSelfAttention(input_dim = input_dim, lstm_hid_dim=lstm_hid_dim, output_dim = output_dim)
    
    output, attention = model(batch_x)