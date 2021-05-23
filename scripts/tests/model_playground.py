import torch
import torch.nn as nn
import model

from importlib import reload  

reload(model)


''' Plays around with Attention dimensionality '''

                   #A #T #G
Q1 = torch.Tensor([[1, 0, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
K1 = torch.Tensor([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
V1 = Q1

                   #A #T #G
Q2 = torch.Tensor([[1, 0, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
K2 = torch.Tensor([[0, 1, 1],
                   [1, 0, 1],
                   [0, 0, 1]])
V2 = Q2

Q = torch.stack([Q1, Q2])
K = torch.stack([K1, K2])
V = torch.stack([V1, V2])

model.attention(Q, K, V)

''' Multi-Head Attention With One-Head '''
QD = torch.stack([Q, Q])
KD = torch.stack([K, K])
VD = torch.stack([V, V])

l = nn.Linear(3, 3)
m = model.MultiHeadedAttention(1, 3, dropout=0)

Q

l(Q)

l(Q1.unsqueeze(0))

l(Q).view(2, -1, 1, 3).transpose(1, 2)

m(Q1.unsqueeze(0), K1.unsqueeze(0), V1.unsqueeze(0))

def reshape_targets(X):
    return X.view(X.size(0) * X.size(1), X.size(2), X.size(3))

# Yippeee: I don't need to change attention mechanism,
# I can just reshape and then un-reshape :)
m(reshape_targets(QD), reshape_targets(KD), reshape_targets(VD))
m(Q, K, V)

QD

None

'''
Decoder Variant of MultiHeadedAttention 

Opting out of this in favor of simply reshaping input and output
tensors completely.
'''
class DecoderMultiHeadedAttention(nn.Module):
    '''
    Generalization of standard attention mechanism to a 4D tensor
    to allow multiple targets per each input.
    '''
    def __init__(self, h, d_model, dropout=0.1):
        super(DecoderMultiHeadedAttention, self).__init__()
        self.attn = model.MultiHeadedAttention(h, d_model, dropout)

    def reshape(self, X):
        return X.view(X.size(0) * X.size(1), X.size(2), X.size(3))

    def forward(self, query, key, value, mask=None):
        "4D variant works by reshaping."
        query = self.reshape(query)
        key = self.reshape(key)
        value = self.reshape(value)

        res = self.attn(query, key, value)
        return 


''' Playing with splat '''

X = torch.stack([Q1, Q2])
Y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
splat(X, Y.size(0) // X.size(0))

''' Simple Model Integration Testing '''

X = ["CACGACTGAAAATCTTATTC",
     "ACAGTCGGTCATATTGGGGT"]

Y = [["CACGACTGAAAATCTTATTC", "CACGACTGAAAATCTTATTC",  "CACGACTGTAAATCTTATTC"],
     ["ACAGTTGGTCATATTGGGGT", "ACAGTCGGTCATATTGGGGT",  "ACAGTTTTTCATATTGGGGT"]]

def embed_input(X):
    return torch.stack([model.one_hot(x) for x in X])

def embed_target(Y):
    t = torch.stack([model.one_hot(t) for TS in Y for t in TS])
    return t.reshape(len(Y), len(Y[0]), t.size(1), t.size(2))

X_e = embed_input(X)
Y_e = embed_target(Y)

m = model.make_model(
    4,
    N=2,
    d_model=X_e.size(-1),
    d_ff=512,
    h=1,
    dropout=0.0,
)
                 
X_e.shape
Y_e.shape

mask = model.subsequent_mask(20)
c = copy.deepcopy
res = m(X_e, Y_e, c(mask), c(mask))

res.shape

assert torch.all(res[0][1] == res[0][0]).item()

''' Testing Loss Function '''

out = torch.tensor(
    [[[[1/2, 1/2, 0,  0],
      [1,   0, 0,    0],
      [0,   0, 9/10, 1/10],
      [0,   1, 0,    0]],

     [[1/4, 0,   0,      3/4],
      [0,   .99, .01,    0],
      [0,   0,   9/10,   1/10],
      [0,   1,   0,      0]]]]
)

target = torch.tensor(
    [[[[1, 0, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 1, 0, 0]],

     [[0, 0, 0, 1],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 1, 0, 0]]]]
)
    

(out + 1e-12).log().mul(target).sum(-1).sum(-1).exp()

# need to use batchmean instead of mean (default)
nn.KLDivLoss(reduction='batchmean')(
    torch.tensor([[1/2, 1/2],
                  [2/3, 1/3]]).log(),
    torch.tensor([[1/4, 3/4],
                  [2/3, 1/3]]))
