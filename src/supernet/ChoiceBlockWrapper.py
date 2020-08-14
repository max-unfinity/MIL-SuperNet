import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import numpy as np
from tqdm.auto import tqdm
import itertools


class ChoiceBlockWrapper(nn.Module):
    '''Модуль, который создает ветвления в SuperNet. При обучении, пути forward выбираются случайно
    (контролируется параметром dropout_gamma). Возможен проход сразу по нескольким параллельным путям в этом блоке,
    результат параллельных путей суммируется и подается на выход. Это ускоряет сходимость.'''
    
    def __init__(self, choice_blocks, dropout_gamma=0.1, skip_chance=0.0, allow_parallel=True):
        '''
        choice_blocks: list(nn.Module). Каждый элемент списка - вариант ветвления.
        dropout_gamma: float. Отвечает за частоту выбрасывания блоков при forward.
            Для каждого блока существует вероятность быть откинутом на этом шаге,
            dropout_prob = dropout_gamma**(1/len(choice_blocks)).
        skip_chance: float. Константная вероятность для skip forward (input'ы просто подаются в output).
        allow_parallel: bool. Допускать ли параллельные пути (не для всех блоков нужна параллельность, например, для активаций).
        '''
        
        super().__init__()
        self.choice_blocks = nn.ModuleList(choice_blocks)
        self.skip_chance = skip_chance
        self.allow_parallel = allow_parallel
        self.set_dropout_gamma(dropout_gamma)
        
        # matmul will be alternative way to apply dropout
        self.register_buffer('dropout_mask', torch.zeros(len(choice_blocks), dtype=torch.int64))
        
        choices_in_size, choices_out_size = self._calc_check_choice_sizes(choice_blocks)
        if skip_chance > 0.:
            assert choices_in_size == choices_out_size, 'if skip_chance == True, choices must have constant channel sizes (in_channels == out_channels)'

    def forward(self, x, force_choice=None, return_choice=False):
        '''force_choice: tuple. Определяет конкретный путь. При None выбирается случайный с np.random.
        Случайность контролируется параметром dropout_gamma
        return_choice: bool. Нужно ли возвращать выбранный путь.'''
        
        x_identity = x
        
        # apply Path-Dropout
        N_choices = len(self.choice_blocks)
        self.dropout_mask.zero_()
        
        if force_choice is None:
            if self.skip_chance > np.random.rand():
                if return_choice:
                    return x_identity, tuple()
                else:
                    return x_identity
            else:
                if self.allow_parallel:
                    self.dropout_mask[np.random.rand(N_choices) > self.dropout_prob] = 1
                if not self.allow_parallel or torch.all(self.dropout_mask == 0):
                    self.dropout_mask[np.random.randint(0, N_choices)] = 1
        else:
            force_choice = list(force_choice)
            self.dropout_mask[force_choice] = 1
            if self.skip_chance == 0. and torch.all(self.dropout_mask == 0):
                import warnings
                warnings.warn(f'can\'t do empty force choice {force_choice} due to skip_chance == 0. Selecting 1 random choice...')
                self.dropout_mask[np.random.randint(0, N_choices)] = 1
                
        # forward choices which has been selected by dropout_mask
        x = [m(x) for m, mask in zip(self.choice_blocks, self.dropout_mask) if mask == 1]
        x = torch.stack(x, dim=-1)  # (B, C, H, W, N_choices)
        x = x.sum(-1)
        
        if return_choice:
            return x, tuple(torch.where(self.dropout_mask)[0].tolist())
        else:
            return x
    
    def set_dropout_gamma(self, gamma):
        self._dropout_gamma = gamma
        self.dropout_prob = self._dropout_gamma**(1/len(self.choice_blocks))
        
    def get_choices(self, parallel_limit=None):
        '''Возвращает все варианты ветвления в этом блоке - комбинация всех путей.
        parallel_limit: int. Ограничение на число параллельных путей. При None, параллельные ветвления не возвратятся.
        return: list(tuple). Список из индексов всех путей, например: [(0,), (1,), (0, 1)]'''
        
        choices = []
        n_blocks = len(self.choice_blocks)
        if parallel_limit is None:
            parallel_limit = n_blocks
        if not self.allow_parallel:
            parallel_limit = 1
        n_choices = min(n_blocks, parallel_limit)
        for i in range(1-int(np.ceil(self.skip_chance)), n_choices+1):
            choices += list(itertools.combinations(range(n_blocks), i))
        return choices
        
    @staticmethod
    def _calc_check_choice_sizes(choice_blocks):
        try:
            in_out_sizes = np.array([list(next(m.parameters()).shape[:2]) for m in choice_blocks])
            assert np.all(in_out_sizes == in_out_sizes[0]), f'some choices don\'t match the same inputs/outputs: {in_out_sizes.T[[1,0]]}'
        except StopIteration:  # m.parameters() can throw this
            return 0, 0
        return in_out_sizes[0, 1], in_out_sizes[0, 0]