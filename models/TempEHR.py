from models.tlstm import TLSTM, StackedTLSTM

class MovingAverage(nn.Module):
    def __init__(self, window_size, hidden_size, input_size, trend_type):
        super(MovingAverage, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.trend_type = trend_type
        self.embedding_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_size * 5, hidden_size)
        self.embed_layer = nn.Linear(input_size, self.embedding_size)

    def exponential_moving_average(self, x):
        alpha = 2 / (x.size(1) + 1) 
        ema = torch.zeros_like(x)
        ema[:, 0, :] = x[:, 0, :]  
        for t in range(1, x.size(1)):
            ema[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        return ema

    def time_exp_moving_average(self, x, timestamps):
        lam = 2 / (x.size(1) + 1)  
        ema = torch.zeros_like(x)
        ema[:, 0, :] = x[:, 0, :]  

        for t in range(1, x.size(1)):
            time_diff = timestamps[:, t] - timestamps[:, t - 1]  
            time_diff = torch.clamp(time_diff, min=1e-6) 
            alpha = torch.exp(-lam / time_diff).view(-1, 1) 
            ema[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * ema[:, t - 1, :] 

        return ema



    def forward(self, x, timestamps):
      
        batch_size, seq_length, input_size = x.size()
        
        if self.trend_type == "ema":
            trend = self.exponential_moving_average(x)
        elif self.trend_type == "tema":
            trend = self.time_exp_moving_average(x, timestamps) 

        out = self.embed_layer(trend)
        return out


class Encoder(nn.Module):    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout=dropout
        self.timeaware = StackedTLSTM(self.input_size, self.hidden_size,self.num_layers)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc_inter = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.LeakyReLU() 
        
    def forward(self, x, time):
        
        hidden,states = self.timeaware(x, time)
        hidden_drp= self.dropout(hidden)
        out= self.fc_inter(hidden_drp)
        return hidden,states

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout= dropout
        self.timeaware = StackedTLSTM(input_size, self.hidden_size, self.num_layers)
        
        self.fc_inter = nn.Linear(hidden_size, hidden_size)
        self.fc_inter2 = nn.Linear(hidden_size, hidden_size)  
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc_time = nn.Linear(hidden_size, 1)
        
        self.relu = nn.LeakyReLU() 
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, time, hidden):
        output, states = self.timeaware(x, time, hidden)
        
        output = self.dropout(output)
        output = self.relu(self.fc_inter(output))
        output = self.fc_inter2(output)
        
        predict_out = self.fc(output)
        time_out = self.fc_time(output)
        
        prediction = self.sigmoid(predict_out)
        time_pred = self.relu(time_out)
        
        return prediction, time_pred, states



class TempEHR(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers_enc,num_layers_dec,num_layers_trend,drp_enc, drp_dec, device=torch.device("cuda")):
        super(TempEHR, self).__init__()
        self.device = device
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.num_layers_trend= num_layers_trend
        self.drp_enc= drp_enc
        self.drp_dec=drp_dec
       
        " lstm ae "
        self.lstm_enc = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers_enc, dropout=drp_enc)
        self.lstm_dec = Decoder(input_size=latent_size, output_size=input_size, hidden_size=hidden_size, num_layers=num_layers_dec, dropout=drp_dec)

        self.fc21 = nn.Linear(self.hidden_size+self.hidden_size,self.latent_size)  
        self.fc22 = nn.Linear(self.hidden_size+self.hidden_size, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.ma= MovingAverage(num_layers_trend, self.hidden_size, self.input_size, 'tema')
        

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, time):
        b, seq_len, dim = x.shape
        
        moving=self.ma(x, time)
        enc_hidden,states= self.lstm_enc(x, time)
        combined = torch.cat((moving, states[0][-1]), dim=-1)
        enc_h=combined[:, -1, :]
        mean = self.fc21(enc_h)
        logvar = self.fc22(enc_h)
        z = self.reparametize(mean, logvar)

        h_ = self.fc3(z)
        z = z.repeat(1, seq_len, 1)
        latent = z.view(b, seq_len, -1)


   
        hidden = (h_,h_)
        x_hat,time_out,dec_states  = self.lstm_dec(latent,time, hidden)
        
        
        return x_hat,time_out, mean, logvar, z, enc_hidden