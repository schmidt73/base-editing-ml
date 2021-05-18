import sys
import torch

sys.path.append('scripts')
import model

outcomes = [
    'ATTC',
    'AGTC',
    'ATGC',
    'ATCC',
    'ATCG',
]

frequencies = torch.tensor([0.5, 0.25, 0.1, 0.1, 0.05])
embedding = torch.stack(list(map(model.one_hot, outcomes)))

model.tensorfy_conditional(embedding, frequencies)
