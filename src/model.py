"""
Implementation of the Bonito model with masked LSTM layers.

Adapted from https://github.com/marcpaga/basecalling_architectures/blob/5db4957496079d19deacb01c9f4f4957f7257f49/models/bonito/model.py
which is based on the Bonito model architecture.

Copyright 2024 Mees Frensel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import torch
from torch import nn

from base_model import BaseModelCRF
from layers.masked_lstm import MaskedLstm

class MaskedModel(BaseModelCRF):
    """Bonito Model adapted with LSTM masking to learn structured sparsity
    """
    def __init__(self, convolution = None, encoder = None, decoder = None, reverse = True, load_default = False, *args, **kwargs):
        super(MaskedModel, self).__init__(*args, **kwargs)
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            reverse (bool): if the first rnn layer starts with reverse 
        """
    
        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        
        if load_default:
            self.load_default_configuration()

    def forward(self, x):
        """Forward pass of a batch
        
        Args:
            x (tensor) : [batch size, channels (1), seq len]
        """

        x = self.convolution(x)
        x = x.permute(2, 0, 1) # [seq len, batch size, channels (=hidden size)]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build_cnn(self):
        return nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5, stride= 1, padding=5//2, bias=True),
            nn.SiLU(),
            nn.Conv1d(in_channels = 4, out_channels = 16, kernel_size = 5, stride= 1, padding=5//2, bias=True),
            nn.SiLU(),
            nn.Conv1d(in_channels = 16, out_channels = 384, kernel_size = 19, stride= 5, padding=19//2, bias=True),
            nn.SiLU()
        )

    def build_encoder(self, input_size, reverse):
        hidden_size = 384

        if reverse:
            encoder = nn.Sequential(MaskedLstm(input_size, hidden_size, reverse = True),
                                    MaskedLstm(hidden_size, hidden_size, reverse = False),
                                    MaskedLstm(hidden_size, hidden_size, reverse = True),
                                    MaskedLstm(hidden_size, hidden_size, reverse = False),
                                    MaskedLstm(hidden_size, hidden_size, reverse = True),
                                    )
        else:
            encoder = nn.Sequential(MaskedLstm(input_size, hidden_size, reverse = False),
                                    MaskedLstm(hidden_size, hidden_size, reverse = True),
                                    MaskedLstm(hidden_size, hidden_size, reverse = False),
                                    MaskedLstm(hidden_size, hidden_size, reverse = True),
                                    MaskedLstm(hidden_size, hidden_size, reverse = False),
                                    )
        return encoder    

    def get_defaults(self):
        defaults = {
            'cnn_output_size': 384, 
            'cnn_output_activation': 'silu',
            'encoder_input_size': 384,
            'encoder_output_size': 384,
            'cnn_stride': 5,
        }
        return defaults
        
    def load_default_configuration(self):
        """Sets the default configuration for one or more
        modules of the network
        """

        self.convolution = self.build_cnn()
        self.cnn_stride = self.get_defaults()['cnn_stride']
        self.encoder = self.build_encoder(input_size = 384, reverse = True)
        self.decoder = self.build_decoder(encoder_output_size = 384)
    
    def _nonzero_cdf(self, log_alpha):
        """Calculate CDF (the probability of s_i being non-zero) element-wise"""
        return torch.sigmoid(torch.add(log_alpha, -MaskedLstm.BETA * math.log(-MaskedLstm.GAMMA / MaskedLstm.ZETA)))

    def train_step(self, batch):
        """Copied from classes.py>BaseModel, and added loss for nonzero s elements.

        Train a step with a batch of data.
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        y = batch['y'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            p = self.forward(x) # forward through the network
            loss, losses = self.calculate_loss(y, p)

        for name, param in self.named_parameters():
            if 'log_alpha_s' in name:
                s_expectation = self._nonzero_cdf(param)

                # Input-to-hidden. When using non-standard model parameters: get 384 from input size directly
                loss += (self.lambda_ * 384 * s_expectation).sum()

                # Hidden-to-hidden, diagonal only
                loss += (self.lambda_ * s_expectation).sum()

                # Hidden-to-hidden, except diagonal
                loss += (self.lambda_ * torch.einsum('i,j->ij', s_expectation, s_expectation)).fill_diagonal_(0).sum()

        self.optimize(loss)
        losses['loss.global'] = loss.item()
        
        return losses, p
