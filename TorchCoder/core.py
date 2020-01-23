# Standard Library
import pandas as pd
import numpy as np

# Third Party
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

# Local Modules
from .autoencoders import RAE, SAE


####################
# Data preparation #
####################
def prepare_data(sequential_data) :
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
        
    seq_len = unsqueezed_data[1]
    no_features = unsqueezed_data[2]
    
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
    temp_final_loss = model.fit(refined_input_data)
    final_loss = temp_final_loss.item()
    
    # recording_results
    embedded_points = model.encode(refined_input_data)
    decoded_points = model.decode(embedded_points)

    return embedded_points, decoded_points, final_loss

if __name__ == "__main__":
    sequences = [[1, 4, 12, 13], [9, 6, 2, 1], [3, 3, 14, 11]]
    encoder, decoder, embeddings, f_loss = QuickEncode(
        sequences,
        embedding_dim=2,
        logging=True
    )

    test_encoding = encoder(torch.tensor([[4.0], [5.0], [6.0], [7.0]]))
    test_decoding = decoder(test_encoding)

    print()
    print(test_encoding)
    print(test_decoding)
