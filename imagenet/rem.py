from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torchvision

import numpy as np
import PIL
from time import time
import logging
import math

import matplotlib.pyplot as plt

class REM(nn.Module):
    def __init__(self, model, optimizer, len_num_keep=0, steps=1, episodic=False, m=0.1, n=3, lamb=1.0, margin=0.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, _, _ = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.m = m
        self.n = n
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb = lamb
        self.margin = margin * math.log(1000)

        self.entropy = Entropy()
        self.tokens = 196
        
    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer):   
        outputs, attn = self.model(x, return_attn=True)
        attn_score = attn.mean(dim=1)[:, 0, 1:]
        len_keeps = []
        outputs_list = []
        
        #######################################################
        ################  M_N = [0, 0.1, 0.2]  ################
        #######################################################
        # len_keep_90 = torch.topk(attn.mean(dim=1)[:, 0, 1:], int(196*0.9), largest=False).indices
        # len_keep_80 = torch.topk(attn.mean(dim=1)[:, 0, 1:], int(196*0.8), largest=False).indices

        # self.model.eval()
        # outputs_fg90 = self.model(x, len_keep=len_keep_90, return_attn=False)
        # outputs_fg80 = self.model(x, len_keep=len_keep_80, return_attn=False)
        # self.model.train()
        
        # entropys = self.entropy(outputs)
        # entropys_fg90 = self.entropy(outputs_fg90)
        # entropys_fg80 = self.entropy(outputs_fg80)

        # loss =  softmax_entropy(outputs_fg90,outputs.detach()).mean()
        # loss += softmax_entropy(outputs_fg80,outputs_fg90.detach()).mean()
        # loss += softmax_entropy(outputs_fg80,outputs.detach()).mean()

        # lossn =  (F.relu(entropys-entropys_fg90.detach()+self.margin)).mean()
        # lossn += (F.relu(entropys_fg90-entropys_fg80.detach()+self.margin)).mean()
        # lossn += (F.relu(entropys-entropys_fg80.detach()+self.margin)).mean()
        #######################################################
        
        self.model.eval()
        for m in self.mn:
            if m == 0.0:
                len_keeps.append(None)
                outputs_list.append(outputs)
            else:
                num_keep = int(self.tokens * (1 - m))
                len_keep = torch.topk(attn_score, num_keep, largest=False).indices
                len_keeps.append(len_keep)
                out = self.model(x, len_keep=len_keep, return_attn=False)
                outputs_list.append(out)
        self.model.train()

        loss = 0.0
        for i in range(1, len(self.mn)):
            loss += softmax_entropy(outputs_list[i], outputs_list[0].detach()).mean()
            for j in range(1, i):
                loss += softmax_entropy(outputs_list[i], outputs_list[j].detach()).mean()

        entropys = [self.entropy(out) for out in outputs_list]
        lossn = 0.0
        for i in range(len(self.mn)):
            for j in range(i + 1, len(self.mn)):
                lossn += (F.relu(entropys[i] - entropys[j].detach() + self.margin)).mean()
            
        loss = loss + self.lamb * lossn
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return outputs


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []

    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, nn.LayerNorm):
           for np, p in m.named_parameters():
               if np in ['weight', 'bias'] and p.requires_grad:
                   params.append(p)
                   #print(nm, np)

    return params


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
        # if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            #m.track_running_stats = True
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
