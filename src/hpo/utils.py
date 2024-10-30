import os
from typing import Dict, List, Union

from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_hparams(self, hparams: Dict, metrics: Dict):
        self.writer.add_hparams(hparams, metrics)

    def close(self):
        self.writer.close()


class HPOParameterSpace:
    def __init__(self, hyp_config: Dict):
        self.hyp_config = hyp_config

    def get_space(self) -> Dict[str, Union[List[float], List[str]]]:
        space = {
            'lr0': [self.hyp_config['lr0']['min'], self.hyp_config['lr0']['max']],
            'momentum': [self.hyp_config['momentum']['min'], self.hyp_config['momentum']['max']],
            'weight_decay': [self.hyp_config['weight_decay']['min'], self.hyp_config['weight_decay']['max']],
            'batch': self.hyp_config['batch']['values'],
            'optimizer': self.hyp_config['hpo_params']['optimizer']['values']
        }
        return space


class CallbackManager:
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event: str, function):
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(function)

    def execute_callbacks(self, event: str, trainer):
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(trainer)
