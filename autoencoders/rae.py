import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .early_stopping import *
from tqdm import tqdm_notebook as tqdm

torch.manual_seed(0)

####################
# LSTM Autoencoder #
####################      
# code inspired by  https://github.com/shobrook/sequitur/blob/master/sequitur/autoencoders/rae.py
# annotation sourced by  ttps://pytorch.org/docs/stable/nn.html#torch.nn.LSTM        

# (1) Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM1(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out
    
# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE(nn.Module):
    def __init__(self, seq_len, no_features, embedding_dim, learning_rate, every_epoch_print, epochs, patience, max_grad_norm):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.every_epoch_print = every_epoch_print
    
    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def fit(self, x):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        criterion = nn.MSELoss(reduction='mean')
        self.train()
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)

        for epoch in range(1 , self.epochs+1):
            # updating early_stopping's epoch
            early_stopping.epoch = epoch        
            optimizer.zero_grad()
            encoded, decoded = self(x)
            loss = criterion(decoded , x)
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(loss, self)
            
            if early_stopping.early_stop:
                break
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
            optimizer.step()
            
            if epoch % self.every_epoch_print == 0:
                print(f"epoch : {epoch}, loss_mean : {loss.item():.7f}")
        
        # load the last checkpoint with the best model
        self.load_state_dict(torch.load('./checkpoint.pt'))
        
        # to check the final_loss
        encoded, decoded = self(x)
        final_loss = criterion(decoded , x).item()
        
        return final_loss
    
    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded
    
    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))
