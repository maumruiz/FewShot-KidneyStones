from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict
from collections import OrderedDict


##########
# Layers #
##########
class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases) -> torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


##########
# Models #
##########
def get_few_shot_encoder(num_input_channels=1) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks

    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64)
    )

class CTMNetwork(nn.Module):
    def __init__(self, k: int, n: int, q: int, num_input_channels: int):
        """Creates a CTM Network

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            num_input_channels: Number of color channels the model expects input data to contain. 
                Omniglot = 1, miniImageNet = 3
            device: Device on which to run computation
        """
        super(CTMNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        self.ctm = CTM(self.k, self.n, self.q)

    def forward(self, inputs):
        pass

class CTM(nn.Module):
    """
    CTM module for few shot segmentation

    Args:
        in_channels:
            number of input channels
    """
    def __init__(self, n_way, n_shot, n_queries, in_channels=64, pretrained_path=None):
        super().__init__()
        
        # TODO: Try other network combinations: Using resnet block, changing d2, d3, m2, m3

        in_concentrator = n_shot * in_channels
        self.concentrator = self._make_vgg_layer(4, in_concentrator, 40) # Input shape (NK, m1, d1, d1) -> Output shape (N, m2, d2, d2)

        in_projector = n_way * 40
        self.projector = self._make_vgg_layer(4, in_projector, 40) # Input shape (1, Nm2, d2, d2) -> Output shape (1, m3, d3, d3)

        self.reshaper = self._make_vgg_layer(4, in_channels, 40) # Output shape (NK, m3, d3, d3)

        self._init_weights()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_queries = n_queries

    def forward(self, supp_fts, query_fts):
        # Support fts: WaxSh, C(64), H'(5), W'(5)
        # Query fts: Q, C(64), H'(5), W'(5)

        # Transform input to correct dimensions: supp(WaxSh, C, H, W) and query(Q, C, H, W)
        fts_size = supp_fts.shape[-2:]
        # supp_fts = batch_supp_fts[:, :, 0].view(-1, batch_supp_fts.size(), *fts_size)
        # query_fts = batch_query_fts[:, :, 0].view(-1, batch_query_fts.size(2), *fts_size)

        ### CONCENTRATOR
        # Reshape to have (n_way, n_shot*in_channels, H, W)
        supp_fts_reshape = supp_fts.view(self.n_way, -1, *fts_size)
        out_concentrator = self.concentrator(supp_fts_reshape)

        ### PROJECTOR
        # Reshape to have (1, n_way*in_channels, H, W)
        input_projector = out_concentrator.view(1, -1, *fts_size)
        out_projector = self.projector(input_projector)
        out_projector = F.softmax(out_projector, dim=1)
        
        ### ENHANCED FEATURES
        out_supp_fts = self.reshaper(supp_fts)
        out_supp_fts = torch.matmul(out_supp_fts, out_projector)

        out_query_fts = self.reshaper(query_fts)
        out_query_fts = torch.matmul(out_query_fts, out_projector)

        # Transform output to correct dimensions: supp(WaxSh, C, H, W) query(Q, C, H, W)
        # out_supp_fts = out_supp_fts.view(self.n_way, self.n_shot, 1, -1, *fts_size)
        # out_query_fts = out_query_fts.view(self.n_queries, 1, -1, *fts_size)

        return out_supp_fts, out_query_fts

    def _make_vgg_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _make_resnet_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Make a resnet layer

        Args:
            blocks:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layers = []
        layers.append(Bottleneck(in_channels, out_channels))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FewShotClassifier(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.logits = nn.Linear(final_layer_size, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""

        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                      weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x


class MatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, fce: bool, num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int, device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(device, dtype=torch.double)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=unrolling_steps).to(device, dtype=torch.double)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise(ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().double()
        c = torch.zeros(batch_size, embedding_dim).cuda().double()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries, (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h
