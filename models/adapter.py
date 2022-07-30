
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN, get_activation


class AdapterModule(nn.Module):
    def __init__(self, input_size, reduction_factor, init_weights="bert", non_linearity="relu"):
        super().__init__()
        self.input_size = input_size
        self.reduction_factor = reduction_factor
        self.down_sample = input_size // reduction_factor
        self.init_weights = init_weights
        self.non_linearity = non_linearity

        seq_list = [nn.Linear(self.input_size, self.down_sample)]
        self.non_linearity = Activation_Function_Class(self.non_linearity)
        seq_list.append(self.non_linearity)
        
        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        if self.init_weights == "bert":
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)
        else:
            raise ValueError("Unknown init_weights type: {}".format(config["init_weights"]))
        
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.adapter_down(hidden_states)
        hidden_states = self.adapter_up(hidden_states)
        return hidden_states + residual
    
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(hidden_act.lower())

    def forward(self, x):
        return self.f(x)


def set_trainable_parameters(module):
    for name, param in module.named_parameters():
        # or "crossattention" in name
        if "adapter" in name or "LayerNorm" in name or "norm" in name or "cls" in name or "pooler" in name or "crossattention" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def print_num_params(model):
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    backbone = sum(p.numel() for name, p in model.named_parameters() if "adapter" not in name)
    print(f"backbone params: {backbone}")
    print(f"total params: {sum(p.numel() for p in model.parameters())}")
    