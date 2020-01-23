# Standard Library
import os
import pandas as pd
import numpy as np

# Third Party
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

# Local Modules
from .autoencoders import LSTM_AE


###############
# GPU Setting #
###############
os.environ["CUDA_VISIBLE_DEVICES"]="0"   # comment this line if you want to use all of your GPUs
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


####################
# Data preparation #
####################
def prepare_dataset(sequential_data) :
    if type(sequential_data) == pd.DataFrame:
        data_in_numpy = np.array(sequential_data)
        data_in_tensor = torch.tensor(data_in_numpy, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == np.array:
        data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == list:
        data_in_tensor = torch.tensor(sequential_data, dtype = torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
        
    seq_len = unsqueezed_data.shape[1]
    no_features = unsqueezed_data.shape[2] 
    # shape[0] is the number of batches
    
    return unsqueezed_data, seq_len, no_features


##################################################
# QuickEncode : Encoding & Decoding & Final_loss #
##################################################
def QuickEncode(input_data, 
                embedding_dim, 
                learning_rate = 1e-3, 
                every_epoch_print = 100, 
                epochs = 10000, 
                patience = 20, 
                max_grad_norm = 0.005):
    
    refined_input_data, seq_len, no_features = prepare_dataset(input_data)
    model = LSTM_AE(seq_len, no_features, embedding_dim, learning_rate, every_epoch_print, epochs, patience, max_grad_norm)
    final_loss = model.fit(refined_input_data)
    
    # recording_results
    embedded_points = model.encode(refined_input_data)
    decoded_points = model.decode(embedded_points)

    return embedded_points.cpu().data, decoded_points.cpu().data, final_loss
