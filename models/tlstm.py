import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


## from https://github.com/duskybomb/tlstm/blob/master/tlstm.py
## https://github.com/yinchangchang/CAT-LSTM/blob/main/code/model.py
class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.Wi = nn.Linear(input_size, hidden_size, bias=True)
        self.Ui = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wf = nn.Linear(input_size, hidden_size, bias=True)
        self.Uf = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wo = nn.Linear(input_size, hidden_size, bias=True)
        self.Uo = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wc = nn.Linear(input_size, hidden_size, bias=True)
        self.Uc = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_decomp = nn.Linear(hidden_size, hidden_size)
        
        
        self.W_t = nn.Linear(1, 1)
        self.b_t = nn.Parameter(torch.zeros(1))


    def map_elapse_time(self, t):
        return 1 / torch.log(t + 2.7183+ 1e-6)

    def forward(self, inputs, timestamps, hx=None):
        batch_size, seq_len, _ = inputs.size()

        if hx is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t, c_t = hx

        outputs = []
        
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            time_t = timestamps[:, t].unsqueeze(1)  
            T = self.map_elapse_time(time_t) 

            C_ST = torch.tanh(self.W_decomp(c_t))  
            C_ST_dis = T * C_ST  
            c_t = c_t - C_ST + C_ST_dis 

            i_t = torch.sigmoid(self.Wi(x_t) + self.Ui(h_t))
            f_t = torch.sigmoid(self.Wf(x_t) + self.Uf(h_t))
            o_t = torch.sigmoid(self.Wo(x_t) + self.Uo(h_t))
            C_t = torch.tanh(self.Wc(x_t) + self.Uc(h_t))
            
            c_t = f_t * c_t + i_t * C_t
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        return outputs, (h_t.unsqueeze(0), c_t.unsqueeze(0))


class StackedTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):

        super(StackedTLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

       
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size  
            self.layers.append(TLSTM(layer_input_size, hidden_size))

    def forward(self, inputs, timestamps, hx=None):
       
        batch_size, seq_len, _ = inputs.size()

        if hx is None:
            h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=inputs.device)
            c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=inputs.device)
        else:
            h_t, c_t = hx

        layer_input = inputs  
        hidden_states = []
        cell_states = []

        for i, layer in enumerate(self.layers):
            output, (h_i, c_i) = layer(layer_input, timestamps, (h_t[i], c_t[i]))
            layer_input = output 
            hidden_states.append(h_i)
            cell_states.append(c_i)

      
        h_n = torch.cat(hidden_states, dim=0)  
        c_n = torch.cat(cell_states, dim=0) 

        return output, (h_n, c_n)

