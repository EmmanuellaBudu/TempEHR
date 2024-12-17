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
    def __init__(self, input_size, hidden_size, cuda_flag=True):
        super(TLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        
        
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        self.Wog = nn.Linear(input_size, hidden_size)
        self.Uog = nn.Linear(hidden_size, hidden_size)
        self.bog = nn.Parameter(torch.zeros(hidden_size))

        self.Wc = nn.Linear(input_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size)
        self.bc = nn.Parameter(torch.zeros(hidden_size))

       
        self.W_decomp = nn.Linear(hidden_size, hidden_size)
        self.b_decomp = nn.Parameter(torch.zeros(hidden_size))

    def map_elapse_time(self, t):
    
        return 1 / torch.log(t + 2.7183) #epsilon value from Baytas implementation 

    def forward(self, inputs, timestamps, initial_states=None):
        b, seq, embed = inputs.size()
        if initial_states is None:
            h = torch.zeros(b, self.hidden_size).to(inputs.device)
            c = torch.zeros(b, self.hidden_size).to(inputs.device)
        else:
            h, c = initial_states
        
        outputs = []
        hidden_states = []  
        cell_states = []   
        
        for s in range(seq):
            x = inputs[:, s, :]
            time_t = timestamps[:, s:s + 1]
            T = self.map_elapse_time(time_t)
            
           
            C_ST = torch.tanh(self.W_decomp(c) + self.b_decomp)
            C_ST_dis = T * C_ST
            c = c - C_ST + C_ST_dis  

           
            i = torch.sigmoid(self.Wi(x) + self.Ui(h) + self.bi)
            f = torch.sigmoid(self.Wf(x) + self.Uf(h) + self.bf)
            o = torch.sigmoid(self.Wog(x) + self.Uog(h) + self.bog)
            C = torch.tanh(self.Wc(x) + self.Uc(h) + self.bc)
            c = f * c + i * C  
            h = o * torch.tanh(c)  
            
            outputs.append(h)
            hidden_states.append(h)  
            cell_states.append(c)    
        
       
        outputs = torch.stack(outputs, 1)
        hidden_states = torch.stack(hidden_states, 1)  
        cell_states = torch.stack(cell_states, 1)      
        return outputs, (hidden_states, cell_states)

class StackedTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cuda_flag=True):
        super(StackedTLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([TLSTM(input_size if i == 0 else hidden_size, hidden_size, cuda_flag) for i in range(num_layers)])
        self.cuda_flag = cuda_flag

    def forward(self, inputs, timestamps, initial_state=None):
        b, seq, _ = inputs.size()
        if initial_state is None:
            initial_states = [
                (torch.zeros(b, self.layers[i].hidden_size).to(inputs.device),
                 torch.zeros(b, self.layers[i].hidden_size).to(inputs.device))
                for i in range(self.num_layers)]
        else:
            initial_states = [initial_state] * self.num_layers

        curr_input = inputs
        states = []
        final_output = None
        
        for i, layer in enumerate(self.layers):
            current_state = initial_states[i]
            output, state = layer(curr_input, timestamps, current_state)
            final_output = output
            curr_input = output
            states.append(state)
        return final_output, states