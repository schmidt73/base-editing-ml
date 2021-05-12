import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from funcy import partition
from torch.autograd import Variable

#####################
##   DNA Encoding  ##
#####################

''' Useful Helper Functions '''
first  = lambda x: x[0]
second = lambda x: x[1]

def random_dna_sequence(k=20):
    return ''.join(random.choices(list("ATCG"), k=k))

'''
Converts a string to a list of kmers of a specified size.  On the
boundaries of the string inserts a specified default character.

For example,
  list(get_kmers('ACGT', k = 3)) == ['BAC', 'ACG', 'CGT', 'GTB']
'''
def get_kmers(seq, k=3, boundary='B'):
    kmers, d = [], (k - 1) // 2
    for i in range(len(seq)):
        if i - d < 0:
            yield (boundary * (d - i)) + seq[:i+d+1]
            continue

        if i + d >= len(seq):
            yield seq[i-d:] + boundary * (i + d + 1 - len(seq))
            continue

        yield seq[i-d:i+d+1]

def gen_kmers(alphabet, k=3):
    stack, kmers = alphabet.copy(), []
    while stack:
        symbol = stack.pop()

        if len(symbol) == k:
            kmers.append(symbol)
            continue

        for letter in alphabet:
            stack.append(symbol + letter)

    return kmers
        
def kmer_encoding(k=3):
    encoding = {}
    kmers = gen_kmers(['A', 'T', 'C', 'G', 'B'], k)

    # Ensures the encoding is a power of 2
    dim = int(2 ** np.ceil(np.log2(len(kmers))))
    for i in range(len(kmers)):
        enc = torch.zeros(dim)
        enc[i] = 1
        encoding[kmers[i]] = enc

    return encoding

# B represents overflow on the boundary
ONE_HOT_MAP = kmer_encoding()
def one_hot(seq):
    enc = list(map(lambda k: ONE_HOT_MAP[k], get_kmers(seq)))
    return torch.stack(enc)

DECODE_MAP  = {v.argmax().item(): k for k, v in ONE_HOT_MAP.items()}
def decode_output(output_tensor):
    values = torch.argmax(output_tensor, dim=-1).flatten().tolist()
    return ''.join([DECODE_MAP[n][1] if n in DECODE_MAP else '?' for n in values])

###################
## Data Batching ##
###################

def tensorfy_target(row):
    ohe = one_hot(row['outcome'])
    n   = row[['count_r1', 'count_r2']].sum()
    d   = row[['total_r1', 'total_r2']].sum()
    return [ohe, n / d]

def tensorfy_targets(targets, max_targets):
    dna_len = len(targets.iloc[0]['outcome'])
    tensors = list(targets.apply(tensorfy_target, axis=1))

    # Ensure that we have the same number of targets across batch
    # by padding with random targets
    for i in range(max_targets - len(targets)):
        tensors.append([one_hot(random_dna_sequence(k=dna_len)), 0])

    embedding_tensor = torch.stack(list(map(first, tensors)))
    frequency_tensor = torch.tensor(list(map(second, tensors)))

    return embedding_tensor, frequency_tensor

def create_batches(df, batch_size=16):
    def create_batch(sgrna_ids):
        samples = df[df['sgrna_id'].isin(set(sgrna_ids))]
        max_targets = samples.groupby('sgrna_id').count().sgrna.max()

        batch = samples.groupby('sgrna_id').apply(
            lambda df: (
                one_hot(df['native_outcome'].iloc[0]),
                tensorfy_targets(df, max_targets)
            )
        )

        inputs = torch.stack(list(batch.apply(first)))
        targets = torch.stack(list(batch.apply(lambda x: x[1][0])))
        frequencies = torch.stack(list(batch.apply(lambda x: x[1][1])))

        return inputs, targets, frequencies

    sgrnas = df['sgrna_id'].sample(df['sgrna_id'].size).unique()
    for batch in partition(16, sgrnas):
        yield create_batch(batch)

###########
## MODEL ##
###########

HYPER_PARAMETERS = {
    'dropout': 0.1,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'nhead': 1,
    'dim_feedforward': 512,
    'activation': 'relu',
}

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def unsplat(self, X):
        return X.view(X.size(0) * X.size(1), X.size(2), X.size(3))

    def decode(self, memory, src_mask, tgt, tgt_mask):
        shp = tgt.shape
        tgt = self.unsplat(tgt)
        out = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return out.view(*shp)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def splat(self, X, k):
        Y = X.unsqueeze(1).expand(X.size(0), k, X.size(1), X.size(2))
        Y = Y.reshape(-1, X.size(1), X.size(2))
        return Y

    def forward(self, x, memory, src_mask, tgt_mask):
        memory = self.splat(memory, x.size(0) // memory.size(0)) # NEW
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        '''
        Q, K, V \in R^B x R^N x R^E where B = batch size
        '''
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def make_model(tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        c(position), c(position),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

#####################
##    Training     ##
#####################

df = pd.read_csv('../data/be-hive/processed/mES_12kChar_BE4_H47ES48A.csv')
res = next(create_batches(df))

print(res[0].shape)
print(res[1].shape)
print(res[2].shape)

if __name__ == "__main__":
    pass
