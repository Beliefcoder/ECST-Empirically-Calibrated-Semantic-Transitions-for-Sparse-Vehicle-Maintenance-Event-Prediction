import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


def load_sparse(path):
    with np.load(path) as packed:
        values = packed['values']
        if values.size and np.all((values == 0) | (values == 1)):
            values = values.astype(np.uint8)
        result = np.zeros(tuple(packed['shape']), dtype=values.dtype)
        result[tuple(packed['idx'])] = values
    return result


def save_sparse(path, values):
    indices = np.where(values > 0)
    np.savez_compressed(path, idx=indices, values=values[indices], shape=values.shape)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    return json.loads(Path(path).read_text())


def load_priors(root):
    train = Path(root) / 'standard/train'
    return tuple(sparse.csr_matrix(load_sparse(train / name)) for name in
                 ['transition_T_decay_multi.npz', 'transition_T_llm.npz'])


class Dataset:
    def __init__(self, path, batch_size, device, shuffle=False):
        path = Path(path)
        self.visits = load_sparse(path / 'code_x.npz')
        self.labels = load_sparse(path / 'code_y.npz')
        self.lengths = np.load(path / 'visit_lens.npz')['lens']
        self.indices = np.arange(len(self.labels))
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __iter__(self):
        for start in range(0, len(self.labels), self.batch_size):
            selected = self.indices[start:start + self.batch_size]
            yield (torch.as_tensor(self.visits[selected], device=self.device),
                   torch.as_tensor(self.lengths[selected], dtype=torch.long, device=self.device),
                   torch.as_tensor(self.labels[selected], dtype=torch.float32, device=self.device))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)