from models.tlstm import TLSTM, StackedTLSTM

class MovingAverage(nn.Module):
    def __init__(self, window_size, hidden_size, input_size, trend_type):
        super(MovingAverage, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.trend_type = trend_type
        self.embedding_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        
        self.embed_layer = nn.Linear(input_size, self.embedding_size)

    def exponential_moving_average(self, x):
        alpha = 2 / (x.size(1) + 1)
        ema = torch.zeros_like(x).to(x.device)
        ema[:, 0, :] = x[:, 0, :]
        for t in range(1, x.size(1)):
            ema[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        return ema

    def time_exp_moving_average(self, x, timestamps):
        lam = 1 / (x.size(1) + 1)
        ema = torch.zeros_like(x).to(x.device)
        ema[:, 0, :] = x[:, 0, :]
        
        for t in range(1, x.size(1)):
            time_diff = timestamps[:, t] 
            time_diff = torch.clamp(time_diff, min=1e-6) 
            alpha = torch.exp(-lam * (1 / time_diff)).view(-1, 1)
            ema[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        
        return ema

    def forward(self, x, timestamps):
        batch_size, seq_length, input_size = x.size()
        
        if self.trend_type == "ema":
            trend = self.exponential_moving_average(x)
        elif self.trend_type == "tema":
            trend = self.time_exp_moving_average(x, timestamps)
        else:
            raise ValueError(f"Unsupported trend_type: {self.trend_type}")
        
     
        out = torch.relu(self.embed_layer(trend))
        return out

    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.timeaware = StackedTLSTM(input_size, hidden_size, num_layers)
        self.fc_inter = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x, timestamps):
        hidden, states = self.timeaware(x, timestamps)
        return hidden, states


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.timeaware = StackedTLSTM(input_size, hidden_size, num_layers)
        self.fc_inter = nn.Linear(hidden_size, hidden_size)
        self.fc_inter2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fc_time = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(output_size, input_size)  
        
    def forward(self, x, timestamps, hidden, autoregressive=False):
        batch_size, seq_len, _ = x.size()
        outputs = []
        time_preds = []

        x_t = x[:, 0, :].unsqueeze(1)  

        for t in range(seq_len):
            output, hidden = self.timeaware(x_t, timestamps[:, t].unsqueeze(1), hidden)
            output = self.dropout(output)

            prediction = torch.sigmoid(self.fc(output))  
            time_pred = torch.relu(self.fc_time(output))

            outputs.append(prediction)
            time_preds.append(time_pred)

            if autoregressive:
                x_t = self.projection(prediction) 
            else:
                if t + 1 < seq_len:
                    x_t = x[:, t + 1, :].unsqueeze(1)  

        outputs = torch.cat(outputs, dim=1)
        time_preds = torch.cat(time_preds, dim=1)

        return outputs, time_preds, hidden
    

class LSTMVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers_enc, num_layers_dec, num_layers_trend, drp_enc, drp_dec, device):
        super(LSTMVAE, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers_trend = num_layers_trend
        self.lstm_enc = Encoder(input_size, hidden_size, num_layers_enc, drp_enc)
        self.lstm_dec = Decoder(latent_size, hidden_size, input_size, num_layers_dec, drp_dec)
        self.fc21 = nn.Linear(hidden_size + hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size + hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.relu = nn.ReLU()
        self.ma = MovingAverage(num_layers_trend, self.hidden_size, self.input_size, 'tema')
        self.timeaware = StackedTLSTM(hidden_size, hidden_size, num_layers=2)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, time, autoregressive=False):
        enc_hidden, (hn, cn) = self.lstm_enc(x, time)
        moving = self.ma(x, time)
        mv_hidden, (h, c) = self.timeaware(moving, time)

        combined = torch.cat((hn[-1], h[-1]), dim=-1)
        mean = self.fc21(combined)
        logvar = self.fc22(combined)
        z = self.reparametrize(mean, logvar)

        h_ = self.fc3(z)
        h_s = h_.unsqueeze(0).repeat(self.lstm_dec.num_layers, 1, 1)
        latent = z.unsqueeze(1).expand(-1, x.size(1), -1)

        hidden = (h_s, h_s)

        x_hat, time_out, _ = self.lstm_dec(latent, time, hidden, autoregressive=autoregressive)

        return x_hat, time_out, mean, logvar, z, enc_hidden
