import torch
import numpy as np
import math
from torch.nn import functional as F

"""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""
""""""" Modules used in wavenet """""""
"""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""
    
class CausalConv1d(torch.nn.Module):
    """
    Causal Convolution:
        INPUT: (N x C_in x L) tensor 
            N: Minibatch size
            C_in: input channels
            L: input (and output) length
        HYPERPARAMETERS:
            in_channels: number of input channels
            out_channels: number of output channels
        LEARNABLE PARAMETERS: (without bias)
            conv layer: 2 x C_in x C_out
        OUTPUT: (N x C_out x L) tensor
            C_out: output channel
            
        ARCHITECTURE:
            
              q0 q2 q3 ...  qn
              .  .  .       .
              .  .  .       .
              .  .  .       .
              p0 p2 p3 ...  pn
              o0 o2 o3 ...  on
             /| /| /|  ... /|  
            0 i1 i2 i3 ...  in
            0 j1 j2 j3 ...  jn
            . .  .  .       .
            . .  .  .       .
            . .  .  .       .
            0 k1 k2 k3 ...  kn
    """
    
    def __init__(self, in_channels, out_channels, bias = False):
        super(CausalConv1d, self).__init__()

        # padding=1 helps with L_in = L_out = L
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=2, stride=1, 
                                    padding=1, bias=bias)

    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        output = self.conv(x)
        
        # remove last value to enforce L_in = L_out = L
        return output[:, :, :-1]
    
class CausalConvReluPool(torch.nn.Module):
    """
    Combination of three layers
        INPUT: (N x C_in x L_in) tensor
            N: Minibatch size
            C_in: Number of input channels
            L_in: Length of input sequence
        HYPERPARAMETERS:
            pooling_kernel: size of pooling window
        LEARNABLE PARAMETERS: (without bias)
            conv: 2 x C_in x C_out
        OUTPUT: (N x C_out) tensor
            C_out: Number of output channels
            
        ARCHITECTURE:
        
            Causal Conv---ReLU---Average Pool
    """
    def __init__(self, in_channels, out_channels, pooling_kernel, bias = False):
        super(CausalConvReluPool, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, bias = bias)
        self.ReLU = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool1d(kernel_size = pooling_kernel)
        
    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        output = self.conv(x)
        output = self.ReLU(output)
        output = self.pool(output)
        return output
    
class DilatedCausalConv1d(torch.nn.Module):
    """
    Dilated Causal Convolution
        INPUT: (N x C_in x L_in) tensor 
            N: Minibatch size
            C_in: input channels
            L: input length
        HYPERPARAMETERS:
            in_channels: number of input channels
            out_channels: number of output channels
            dilation: spacing between kernel points
        LEARNABLE PARAMETERS: (without bias)
            conv_layer: 2 x C_in x C_out
        OUTPUT: (N x C_out x L_out) tensor
            C_out: output channel
            L_out: output length
            
                    L_out = L_in - dilatation
                    
        ARCHITECTURE:
                
                    q_{j+d}
                    .
                    .
                    .
                    p_{j+d}
                    o_{j+d}  j = 0, ..., L_in - d
                    
                /     |
                
            i_j ... i_{j+d}  j = 0, ..., L_in - d
            k_j ... j_{j+d}
            .       .
            .       .
            .       .
            l_j     l_{j+d}
            
        REMARKS:
            * d = 1 gives Causal convolution without the first entry
                    
    """
    def __init__(self, in_channels, out_channels, dilation=1, bias = False):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels = in_channels, 
                                    out_channels = out_channels,
                                    kernel_size=2, 
                                    stride=1,  
                                    dilation=dilation,
                                    padding=0, 
                                    bias=bias)

    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        output = self.conv(x)
        return output
    
class ResidualBlock(torch.nn.Module):
    """
    Residual block
        INPUT: (N x res_channels x L_in) tensor
            N: Size of minibatch
            L_in: Length of input
        HYPERPARAMETERS:
            res_channels: number of residual channels for input, output
            skip_channels: number of skip channel for skip-output
            dilation: spacing between kernel points
            pooling_kernel: size of kernel for average pooling
        LEARNABLE PARAMETERS: (without bias)
            dilated layer: 2 x res_channels ** 2
            conv_res layer: res_channels ** 2
            conv_skip layer: res_channels * skip_channels
            TOTAL: res_channels * (3 * res_channels + skip_channels)
        OUTPUT: 
            residual_output: (N x RC x L_out) tensor:  
                L_out: floor((L_in - dilation) /  pooling_kernel)
            skip_output: (N x skip_channels x skip_size) tensor
            
        ARCHITECTURE:
        
        
             Res output
                  |
        ---------Add
        |         | 
        |     1 x 1 conv
        |         |
        |    Mean pooling----1 x 1 conv----skip output
        |         |
        |      Multiply
        |     /        \
        |  tanh       sigmoid  
        |     \        /
        |    Dilated Conv
        |         |
        -------Input
        
        REMARKS:
            * Note that the number of channels does not change towards the main 
              output, only towards the skip connection.
    """
    def __init__(self, res_channels, skip_channels, 
                 dilation = 1, pooling_kernel = 1, bias = False):
        super(ResidualBlock, self).__init__()
        

        self.dilated = DilatedCausalConv1d(res_channels,
                                           res_channels,
                                           dilation=dilation,
                                           bias = bias)
        
        self.pooling = False
        if pooling_kernel > 1:
            self.mean_pool = torch.nn.AvgPool1d(kernel_size = pooling_kernel)
            self.pooling = True
        
        self.conv_res = torch.nn.Conv1d(res_channels, 
                                    res_channels, 
                                    kernel_size = 1,
                                    bias = bias)
        
        self.conv_skip = torch.nn.Conv1d(res_channels, 
                                    skip_channels, 
                                    kernel_size = 1,
                                    bias = bias)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        
        dilat_output = self.dilated(x) # Out: N x RC x (L_in - dilation)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(dilat_output)
        gated_sigmoid = self.gate_sigmoid(dilat_output)
        gated = gated_tanh * gated_sigmoid
        
        # Mean pooling
        if self.pooling > 1:
            pool_out = self.mean_pool(gated)
        else:
            pool_out = gated

        # Residual network
        conv_res_output = self.conv_res(pool_out)
        input_cut = x[:, :, -pool_out.size(2):] 
        res_output = conv_res_output + input_cut

        # Skip connection
        skip_output = self.conv_skip(pool_out)

        return res_output, skip_output
    
class ResidualStack(torch.nn.Module):
    """
    Stack of residual blocks
        INPUT: (N x res_channels x L_in) tensor
            N: Minibatch size
            L_in: Length of input
        HYPERPARAMETERS:
            layer_size: Size of substacks with increasing dilation
            stack_size: Number of substacks
            res_channels: Number of residual channels for input, output
            skip_channels: Number of channels for skip-output
            pooling_kernel: Size of pooling in last block
        LEARNABLE PARAMETERS: (without bias)
            Each block: res_channels * (3 * res_channels + skip_channels)
            Total number of blocks: layer_size * stack_size
            TOTAL: layer_size * stack_size * res_channels
                        * (3 * res_channels + skip_channels)
        OUTPUT: 
            residual_output: (N x res_channels x L_out) tensor
            skip_output: (N x skip_channels x L_out) tensor
            
            L_out = floor(L_in - stack_size * (2 ** layer_size - 1) / pooling_kernel)
        
        ARCHITECTURE:
    
                   Res_out        
                      |
        ResBlock layer_size * stack_size d =  2 ** layer_size, pool- 
                      |                                             |
                      .                                             |
                      .                                             |
                      .                                             |
                      |                                             |
            ResBlock layer_size + 1, d = 1 -------------------------
                      |                                             + ----- Skip_out
        ResBlock layer_size , d = 2 ** layer_size ------------------
                      |                                             |
                      .                                             |
                      .                                             |
                      .                                             |
                      |                                             |
                  ResBlock 2, d=2 ----------------------------------|
                      |                                             |
                  ResBlock 1, d=1 ----------------------------------
                      |
                    Input
            
    Note that the total number of residual blocks is layer_size * stack_size.
    """
    def __init__(self, layer_size, stack_size, res_channels, skip_channels,
                pooling_kernel = 1, bias = False):
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size
        self.skip_channels = skip_channels
        self.pooling_kernel = pooling_kernel
        self.res_blocks = self.stack_res_block(res_channels, skip_channels, 
                                               pooling_kernel)

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation, 
                        pooling_kernel=1, bias = False):
        """ 
        Creates a residual block, parallelizes it and puts it on GPU 
        """
        block = ResidualBlock(res_channels, skip_channels, dilation, 
                              pooling_kernel, bias = bias)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block

    def build_dilations(self):
        """ 
        Build sequence of dilations, stack_size blocks of
        
               1, 2, 4, ... , 2 ** (layer_size - 1)
               
        """
        dilations = []

        for layer in range(0, self.stack_size):
            for level in range(0, self.layer_size):
                dilations.append(2 ** level)

        return dilations

    def stack_res_block(self, res_channels, skip_channels, 
                        pooling_kernel=1, bias = False):
        """
        Builds the stack of stack_size * layer_size residual blocks
        """
        res_blocks = []
        dilations = self.build_dilations()

        for i, dilation in enumerate(dilations):
            if i != len(dilations)-1:
                block = self._residual_block(res_channels, skip_channels, 
                                             dilation, bias = bias)
            else:
                block = self._residual_block(res_channels, skip_channels, dilation, 
                                             pooling_kernel = pooling_kernel,
                                             bias = bias)
            res_blocks.append(block)

        return res_blocks

    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        N = x.shape[0]
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            output, skip = res_block(output)
            skip_connections.append(skip)
        
        osize = output.size(2)
        skip_connections = [s[:,:,-osize:] for s in skip_connections]
        
        skip_out = torch.zeros(skip_connections[0].shape)
        
        for skip in skip_connections:
            skip_out += skip
        
        return output, skip_out
    
class TopLayerGenerator(torch.nn.Module):
    """
    The last couple of layers of the Wavenet's generator part
        INPUT: (N, channels, L) tensor
            N: Minibatch size
            channels: Number of input and output channels
            L: Size of input and output
        LEARNABLE PARAMETERS: (without bias)
            conv1: channels ** 2 parameters
            conv2: channels ** 2 parameters
            TOTAL: 2 * channels ** 2 parameters
        OUTPUT: (N, channels, L) tensor
        
        ARCHITECTURE:
        
          input---relu---conv---relu---conv---softmax
        
    """
    def __init__(self, channels):
        super(TopLayerGenerator, self).__init__()
        
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size = 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size = 1)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.softmax(output)
        return output
    
class TopLayerClassifier(torch.nn.Module):
    """
    The last couple of layers of the Wavenet's classifier part
        INPUT: (N, C_in, L_in) tensor
            N: Minibatch size
            C_in: Number of input and output channels
            L: Size of input and output
        HYPERPARAMETERS:
            mid_channels: Number of intermediate layer channels
            pooling_kernel: Size of pooling window
        LEARNABLE PARAMETERS: (without bias)
            conv1: in_channels * mid_channels parameters
            conv2: mid_channels * classes parameters
            TOTAL: mid_channels(in_channels + classes) parameters
        OUTPUT: (N, classes) tensor
        
        ARCHITECTURE:
        
          AvgPool---CausalConvReluPool---CausalConvReluPool---SoftMax--AvgPool
        
    """
    def __init__(self, in_channels, mid_channels, 
                 classes, pooling_kernel, bias = False):
        super(TopLayerClassifier, self).__init__()
        
        self.pool1 = torch.nn.AvgPool1d(kernel_size = pooling_kernel)
        self.CCRP1 = CausalConvReluPool(in_channels, mid_channels, 
                                        pooling_kernel, bias = bias)
        self.CCRP2 = CausalConvReluPool(mid_channels, classes, 
                                        pooling_kernel, bias = bias)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        assert torch.is_tensor(x), "Tensor input expected"
        assert x.dim() == 3, "Wrong input shape"
        output = self.pool1(x)
        output = self.CCRP1(output)
        output = self.CCRP2(output)
        output = self.softmax(output)
        output = torch.mean(output, dim = 2)
        return output
    
    
class WaveNet(torch.nn.Module):
    """
    A WaveNet for classification:
        INPUT: (N x C_in x L_in)
            N: Minibatch size
            C_in: Number of input channels
            L_in: Input length of waveform
        HYPERPARAMETERS:
            layer_size: Size of substacks with increasing dilation
            stack_size: Number of substacks
            res_channels: Number of channels through residual layers
            skip_channels: number of prediction channels
        LEARNABLE PARAMETERS (without bias)
            CausalConv: 2 * 
        OUTPUT: 
            Class output: (N x classes) tensor
                classes: Number of classes 
            C_out: (N x C_out x L_out) tensor
                C_out: Number of output channels to predict over
                L_out: Predicted output size

                    L_out = L_in - stack_size * (2 ** layer_size - 1)

        ARCHITECTURE:

             Classifier output
                    |
            TopLayerClassifier
                    |                
                 ResStack---------------TopLayerGenerator---Predictor output
                    |
                CausalConv
                    |
                  Input

    At test time, only the vertical part of the architecture is active.

    """
    def __init__(self, layer_size, stack_size, in_channels, 
                 res_channels, skip_channels, classes, 
                 pooling_kernel, bias = False):
        super(WaveNet, self).__init__()
        self.mode = 'train'
        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)
        self.skip_channels = skip_channels
        self.causal = CausalConv1d(in_channels, res_channels, bias = bias)
        self.res_stack = ResidualStack(layer_size, stack_size, 
                                       res_channels, skip_channels)
        self.generator = TopLayerGenerator(skip_channels)
        self.classifier = TopLayerClassifier(res_channels, int(res_channels/2), 
                                             classes, pooling_kernel)
        
    def test(self):
        self.mode = 'test'
        
    def train(self):
        self.mode = 'train'

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        """ Calculates the size of the receptive field for prediction """
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)
        return int(num_receptive_fields)

    def calc_output_size(self, x):
        """ Calculates the output size of the predicting network"""
        output_size = int(x.size(2)) - self.receptive_fields
        self.check_input_size(x, output_size)
        return output_size

    def check_input_size(self, x, output_size):
        """ Checking whether downsampling went too far """
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)

    def forward(self, x):
        causal_out = self.causal(x)
        res_out, skip_out = self.res_stack(causal_out)
        clss = self.classifier(res_out)
        
        if self.mode == 'train':
            pred = self.generator(skip_out)
            return clss, pred
        else:
            return clss
        
"""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""
"""""""""" Loss for wavenet """""""""""
"""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""

def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/
                        pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized

def mu_law_decode(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/
                        pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform

def WaveNetLoss(audios, labels, class_probs, pred_probs, 
                pred_channels, pred_to_class_ratio = 1):
    """
    Computes loss for the Wavenet model
        INPUT: 
            audios: (N x L_in) tensor
                N: Minibatch size
                L_in: Length of input audio
            labels: (N) tensor
            class_probs: (N, 2) tensor
            pred_probs: (N, L_out)
                L_out: Length of predicted outputs
            
        ARCHITECTURE:
            
            InputAudio               PredClass  
                |                        |
             muEncoder---PredProbs     Loss2--------GTClass
                       |                 |
                     Loss1-----------Total Loss
    """
    
    # Encoding the audio and computing skip loss
    N = audios.shape[0]
    pred_channels = pred_channels
    encoded = torch.Tensor(mu_law_encode(audios, pred_channels)).long()
    encoded = encoded[:,-pred_probs.shape[2]:].squeeze()
    
    loss1 = F.cross_entropy(pred_probs, encoded)
    
    # Classification loss
    labels.long()
    loss2 = F.cross_entropy(class_probs,labels)
    
    return pred_to_class_ratio * loss1 + loss2
    
    
    

