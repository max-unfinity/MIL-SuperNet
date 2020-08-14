import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import numpy as np
from tqdm.auto import tqdm
import itertools
from .ChoiceBlockWrapper import ChoiceBlockWrapper


class SuperNet(nn.Module):
    '''Модуль, который строит СуперСеть. Контролирует ветвления, созданные в объектах класса ChoiceBlockWrapper.'''
    
    def __init__(self, modules):
        super().__init__()
        self.net = nn.Sequential(*modules)
        self.total_choices = len(self.get_all_choices())
    
    def forward(self, x, force_choices=None):
        choices = []
        choice_counter = 0
        for m in self.net.children():
            if hasattr(m, 'choice_blocks'):
                force_c = None if force_choices is None else force_choices[choice_counter]
                x, c = m(x, force_c, return_choice=True)
                choices.append(c)
                choice_counter += 1
            else:
                x = m(x)
        return x, tuple(choices)

    def detach_subnet(self, choices=None, copy_weights=True, seed=None):
        '''Создает конкретную subnet, определенную параметром choices.
        Если choices==None, сэмплирует случайную.
        Если copy_weights==False, то созданная subnet будет шарить общие веса с supernet'''
        
        from copy import deepcopy
        if seed is not None: np.random.seed(seed)
        subnet = []
        choice_counter = 0
        for m in self.net.children():
            if hasattr(m, 'choice_blocks'):
                i = choices[choice_counter][0] if (choices is not None) else np.random.randint(0, len(m.choice_blocks))
                m = m.choice_blocks[i]
                choice_counter += 1
            if copy_weights:
                m = deepcopy(m)
            subnet.append(m)
        return nn.Sequential(*subnet)
    
    def get_all_choices(self, parallel_limit=None):
        all_choices = []
        for m in self.net.children():
            if hasattr(m, 'choice_blocks'):
                all_choices.append(m.get_choices(parallel_limit))
        return list(itertools.product(*all_choices))
		
		
class GammaScheduler:
    '''Линейно изменяет dropout_gamma на каждом шаге, с gamma_start до gamma_end'''
    def __init__(self, supernet, total_steps, gamma_start=0.0, gamma_end=0.1):
        self.supernet = supernet
        self.total_steps = total_steps
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.i = -1
        self.step()  # init dropout_gamma in every choice_block
    
    def step(self):
        if self.i >= self.total_steps:
            import warnings
            warnings.warn('GammaScheduler step is out of total_steps. skipping')
            return
        self.i += 1
        for m in self.supernet.net.children():
            if hasattr(m, 'choice_blocks'):
                gamma = self.gamma_start + (self.gamma_end-self.gamma_start)*self.i/self.total_steps
                m.set_dropout_gamma(gamma)               