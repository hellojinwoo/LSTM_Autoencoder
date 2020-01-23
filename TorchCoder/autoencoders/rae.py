import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

torch.manual_seed(0)

###############
# GPU Setting #
###############
os.environ["CUDA_VISIBLE_DEVICES"]="0"   # comment this line if you want to use all of your GPUs
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


##################
# Early Stopping #
##################
# source : https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta= -0.00001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f"Early Stopping activated. Final validation loss : {self.val_loss_min:.7f}")
                self.early_stop = True
        # if the current score does not exceed the best scroe, run the codes following below
        else:  
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), './checkpoint.pt')
        self.val_loss_min = val_loss



class Encoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.num_features))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)

        return hidden_n.reshape((1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.output_dim = 2 * input_dim, output_dim

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # self.perceptrons = nn.ModuleList()
        # for _ in range(seq_len):
        #     self.perceptrons.append(nn.Linear(self.hidden_dim, output_dim))

        self.dense_layers = torch.rand(
            (self.hidden_dim, output_dim),
            dtype=torch.float,
            requires_grad=True
        )

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        # output_seq = torch.empty(
        #     self.seq_len,
        #     self.output_dim,
        #     dtype=torch.float
        # )
        # for index, perceptron in zip(range(self.seq_len), self.perceptrons):
        #     output_seq[index] = perceptron(x[index])
        #
        # return output_seq

        return torch.mm(x, self.dense_layers)


#########
# EXPORTS
#########


class RAE(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim=64):
        super(RAE, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, num_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
