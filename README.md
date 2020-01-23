# TorchCoder

__`TorchCoder`__ is a PyTorch based autoencoder for sequential data, currently supporting only Long Short-Term Memory(LSTM) autoencoder. It is easy to configure and only takes one line of code to use.

## Example
```python
from TorchCoder import *

sequences = [[1,2,3,4],
             [5,6,7,8], 
             [9,10,11,12]]
encoded, decoded, final_loss  = QuickEncode(sequences, embedding_dim=2)

encoded
>> tensor([[ 0.9282,  0.1928],
           [ 0.4378,  0.9898],
           [-0.9958,  0.9990]])
decoded
>> tensor([[ 1.0006,  1.9668,  2.9818,  3.9702],
           [ 4.9706,  6.0296,  6.9568,  7.9714],
           [ 8.9111, 10.6770, 11.0999, 11.2257]])
final_loss
>> 0.09021145105361938
```

`TorchCoder` will learn how to represent sequences of any length in lower-dimensional, fixed-size vectors. This non-linear dimensionality reduction algorithm can be useful for finding patterns among sequences, clustering, and converting sequences into inputs for a machine learning algorithm.

## API

> __`TorchCoder.QuickEncode(input_sequences, embedding_dim, learning_rate, every_epoch_print, epochs, patience, max_grad_norm)`__

Lets you train an autoencoder with just one line of code. This wraps a PyTorch implementation of an Encoder-Decoder architecture with an LSTM, making this optimal for sequences with long-term dependencies (e.g. time series data).

**Parameters**

- `input_sequences`: A list (or tensor) of shape `[num_seqs, seq_len, num_features]` representing your training set of sequences.
  - Each sequence should have the same length, `seq_len`, and contain a sequence of vectors of size `num_features`.
  - If `num_features=1`, then you can input a list of shape `[num_seqs, seq_len]` instead.
  - __[Notice]__ Currently TorchCoder can take `[num_seqs, seq_len]` as an input. Soon to be fixed.
- `embedding_dim`: Size of the vector encodings you want to create.
- `learning_rate`: Learning rate for the autoencoder.
- `every_epoch_print` : Deciding the size of N to print the loss every N epochs.
- `epochs`: Total number of epochs to train for.
- `patience` : Number of epochs to wait for if the loss does not decrease.
- `max_grad_norm` : Maximum size for gradient used in gradient descent (gradient clipping). 

**Returns**

- `encoded`: The encoded vector (representation) of the `input_sequences`.
- `decoded`: The decoded vector of encoded, which should be very close to the `input_sequences`.
- `final_loss`: The final mean squared error of the autoencoder on the training set.

## Contributing

`QuickEncode` is useful for rapid prototyping but doesn't give you much control over the model and training process. For that, you can import the RAE implementation itself from `TorchCoder.autoencoders`.

<!--Provide proof that it's generally effective-->

<!-- https://github.com/szagoruyko/pytorchviz
https://github.com/RobRomijnders/AE_ts
https://github.com/erickrf/autoencoder
https://miro.medium.com/max/1400/1*sWc8g2yiQrOzntbVeGzbEQ.png
https://arxiv.org/pdf/1502.04681.pdf -->
