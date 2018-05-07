import torch.nn as nn
import torch

class ConvLSTM(nn.Module):
    """Convolutional lstm model to classify raw waveform files. The
    architecture is:
    
    Input -> Conv -> ReLU -> Pooling -> LSTM -> Softmax
    
    Properties:
      CONVOLUTIONAL LAYER:
        conv_kernel_size :  The size of the filters for the conv. layer
        conv_stride: Stride for the conv. layer
        num_features: Number of output channels for conv. layer
      MAX POOLING LAYER:
        pooling_kernel: Size of the max-pooling window
      LSTM:
        hidden_size: The dimension of the hidden state
        num_layers: Number of hidden layers inside lstm
        num_of_classes: Number of classes to classify into
        bias: Have/not have bias terms in the layers
    """                     
    
    def __init__(self, conv_kernel_size, conv_stride, num_features, 
                 pooling_kernel, hidden_size, num_layers = 1, 
                 num_of_classes = 2, bias = True):
        super(ConvLSTM, self).__init__()
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.num_features = num_features
        self.pooling_kernel = pooling_kernel
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_of_classes = num_of_classes
        self.bias = bias
        
        # initialize cuda option
        dtype = torch.DoubleTensor # data type
        ltype = torch.LongTensor # label type
        
        self.conv = nn.Conv1d(
            in_channels = 1,
            out_channels = num_features,
            kernel_size = conv_kernel_size,
            stride = conv_stride,
            bias = bias
            )
        

        self.relu = nn.ReLU()
        
        if pooling_kernel > 1:
            self.pooling = nn.MaxPool1d(
                kernel_size = pooling_kernel
                )
        
        self.lstm = nn.LSTM(
            input_size = num_features,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias
            )
        
        self.linear = nn.Linear(
            in_features = hidden_size,
            out_features = num_of_classes,
            bias = bias
            )
        
        self.softmax = nn.Softmax(dim = 1)
        
        #if torch.cuda.is_available():
        #    self.conv = self.conv.cuda()
        #    self.relu = self.relu.cuda()
        #    self.lstm = self.lstm.cuda()
        #    self.linear = self.linear.cuda()
        #    self.softmax = self.softmax.cuda()
                
        
    def forward(self, wav_minibatch):
        """ Forward pass of the Convolutional LSTM audio based 
        ideology classifying network.
        
        INPUT:
            wav_minibatch: The raw waveform of a spoken word
            hidden_init: Initial hidden state
            cell_init: Initial cell state 
        
        OUTPUT:
            prob_score: Probability scores over the classes"""
        
        x = self.conv(wav_minibatch)
        x = self.relu(x)
        if self.pooling_kernel > 1:
            x = self.pooling(x)
        x = x.permute(2,0,1)
        output, x = self.lstm(x)
        x = self.linear(x[0][0,:,:])
        probs = self.softmax(x)
        return probs