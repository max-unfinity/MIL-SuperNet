import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import numpy as np
from tqdm.auto import tqdm
import itertools


class ThompsonSamplingController(nn.Module)
    '''В данный момент просто собирает статистику обучения каждой подсети, чтобы выбрать лучшую.
    Про сам метод ThompsonSampling, я написал в основном ноутбке, в разделе further experimetns'''
    def __init__(self, supernet, parallel_limit=None, is_minimize_score=True)
        # TODO parallel_limit (while training all choices are possible, but for top sampling are not)
        super().__init__()
        self.supernet = supernet
        self.parallel_limit = parallel_limit
        self.is_minimize_score = is_minimize_score
        all_choices = self.supernet.get_all_choices(self.parallel_limit)
        self.subnets_scores = dict(zip(all_choices, [[] for x in range(len(all_choices))]))
        
    def update_score(self, choice, score)
        t = self.subnets_scores.get(choice, None)
        if t is not None
            t.append(score)
            
    def get_scores(self, last_n=None)
        if last_n is None
            last_n = 0
        return {k  np.mean(x[-last_n]) for k, x in list(self.subnets_scores.items()) if len(x)  0}
        
    def get_top_k_choices(self, k=1, last_n=None)
        scores = self.get_scores()
        if len(scores) == 0
            return []
        idxs = np.argsort(list(scores.values()))
        idxs = idxs[k] if self.is_minimize_score else idxs[-1][k]
        l = np.array(list(scores.items()))
        return l[idxs].tolist()