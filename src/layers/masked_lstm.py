"""
This file contains the implementation of an LSTM cell & layer that is masked
with learnable mask s.

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
from typing import List, Tuple

import torch
import torch.jit as jit
from torch import Tensor
from torch.nn.parameter import Parameter

def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]

class MaskedLstmCell(jit.ScriptModule): # type: ignore
    """
    LSTM implementation that adds a selection mechanism in front of the hidden
    neurons, with learnable parameters to 'learn structured sparsity'

    The LSTM cell implementation is adapted from PyTorch's custom_lstms benchmark:
    https://github.com/pytorch/pytorch/blob/6beec34b1c6803d5f6648c3cd7c262d6432374c8/benchmarks/fastrnns/custom_lstms.py
    """

    def __init__(self, input_size, hidden_size):
        super(MaskedLstmCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = Parameter(torch.empty(4 * self.hidden_size, self.input_size))
        self.U = Parameter(torch.empty(4 * self.hidden_size, self.hidden_size))
        self.b = Parameter(torch.empty(4 * self.hidden_size))
        self.log_alpha_s = Parameter(torch.empty(self.hidden_size))

        # Hyper-parameters
        self.beta = MaskedLstm.BETA
        self.gamma = MaskedLstm.GAMMA
        self.zeta = MaskedLstm.ZETA

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name == 'log_alpha_s':
                # "log α is initialized with samples from N(1, 0.1)"
                weight.data.normal_(mean=1, std=0.1)
            else:
                weight.data.uniform_(-stdv, +stdv)

    @jit.script_method # type: ignore
    def _smooth_gate(self, log_alpha: Tensor) -> Tensor:
        u = torch.rand_like(log_alpha)
        τ_hat = torch.sigmoid((u.log() - (-u).log1p() + log_alpha) / self.beta)
        τ = τ_hat * (self.zeta - self.gamma) + self.gamma
        return τ.clamp(0, 1)

    @jit.script_method # type: ignore
    def _estimate_final_gate(self, log_alpha: Tensor) -> Tensor:
        s = torch.clamp(torch.sigmoid(log_alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
        THRESHOLD = 0.3
        s[s <= THRESHOLD] = 0
        s[s > THRESHOLD] = 1
        return s

    @jit.script_method # type: ignore
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state

        if self.training:
            s = self._smooth_gate(self.log_alpha_s)
        else:
            s = self._estimate_final_gate(self.log_alpha_s)

        W_hat = self.W * s.repeat(4).unsqueeze(1) # Uses broadcasting semantics to prevent repeating by 384x
        U_hat = self.U * (s.unsqueeze(1) @ s.unsqueeze(0)).repeat(4, 1)

        gates = (
            torch.mm(input, W_hat.t())
            + torch.mm(hx, U_hat.t())
            + self.b * s.repeat(4)
        )

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate) * s
        forgetgate = torch.sigmoid(forgetgate) * s
        cellgate = torch.tanh(cellgate) * s # tanh preserves 0 so can be skipped in theory
        outgate = torch.sigmoid(outgate) * s

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class MaskedLstm(torch.nn.Module):
    BETA = 2/3 # "Try lambda = 2/3" (https://vitalab.github.io/article/2018/11/29/concrete.html)
    GAMMA = -0.1 # https://arxiv.org/pdf/1811.09332
    ZETA = 1.1
    INITIAL_LAMBDA = 0.00000005

    def __init__(self, input_size, hidden_size, reverse=False):
        super(MaskedLstm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reverse = reverse

        self.lstm = MaskedLstmCell(input_size, hidden_size)

        self.inference_lstm = torch.nn.LSTM(input_size, hidden_size)
        # If this errors, your PyTorch version is too old. Comment this line
        # and uncomment the code in BaseModel.load
        self.register_load_state_dict_post_hook(change_lstm_cell)
        self.must_update_inference_lstm = True
        self.mask = torch.ones(hidden_size, dtype=torch.bool, device='cuda')

    def forward(self, input: Tensor) -> Tensor:
        # x: [sequence len, batch size, input size]
        # e.g. [400       , 64        , 384       ]
        batch_size = input.shape[1]

        if not self.training: # inference
            if self.must_update_inference_lstm:
                change_lstm_cell(self)
                self.must_update_inference_lstm = False

            fast_output, _ = self.inference_lstm(input.flip(0) if self.reverse else input)
            tmp = torch.zeros(input.shape[0], batch_size, self.hidden_size, device=input.device) # original hidden size, not the pruned/masked hidden size
            fast_output = tmp.masked_scatter_(self.mask.unsqueeze(0).unsqueeze(1), fast_output) # Broadcasting semantics to scatter each input tensor individually
            fast_output = fast_output.flip(0) if self.reverse else fast_output

            return fast_output

        self.must_update_inference_lstm = True

        h0 = torch.zeros(batch_size, self.hidden_size, device=input.device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=input.device)
        state = (h0, c0)

        inputs = reverse(input.unbind(0)) if self.reverse else input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.lstm(inputs[i], state)
            assert out.count_nonzero() <= self.lstm._estimate_final_gate(self.lstm.log_alpha_s).count_nonzero() * batch_size
            outputs += [out]

        return torch.stack(reverse(outputs) if self.reverse else outputs)#, state

def change_lstm_cell(module: MaskedLstm, _unmappable=None):
    """Update the inference lstm with the mask to reduce the hidden size"""
    s = module.lstm._estimate_final_gate(module.lstm.log_alpha_s)

    # Shrink weight matrices by removing all zero columns & rows
    num_nonzero = int(s.count_nonzero().item())

    mask_W = s.repeat(4).unsqueeze(1).type(torch.bool)
    mask_U = (s.unsqueeze(1) @ s.unsqueeze(0)).repeat(4, 1).type(torch.bool)
    bias = module.lstm.b.masked_select(s.type(torch.bool).repeat(4)).reshape(num_nonzero * 4)

    new_dict = {
        'weight_ih_l0': module.lstm.W.masked_select(mask_W).reshape(num_nonzero * 4, module.lstm.W.shape[1]),
        'weight_hh_l0': module.lstm.U.masked_select(mask_U).reshape(num_nonzero * 4, num_nonzero),
        'bias_ih_l0': bias,
        'bias_hh_l0': torch.zeros_like(bias),
    }
    module.inference_lstm = torch.nn.LSTM(module.input_size, num_nonzero)
    print("Input, hidden size: {}, {}".format(module.input_size, num_nonzero))
    module.inference_lstm.load_state_dict(new_dict)
    module.inference_lstm.to('cuda')
    module.mask = s.type(torch.bool).to('cuda')
