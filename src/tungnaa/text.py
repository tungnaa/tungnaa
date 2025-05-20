import functools as ft
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from transformers import AutoModel, AutoConfig, CanineModel

def tokenize(text:str, end_tokens:Tuple[int,int]|None):
    """
    Returns:
        tokenized text as LongTensor
        representation of tokenized text as a string
        list mapping index in the original text to the tokenized text
    """
    n = len(text)

    start_tok = [end_tokens[0]] if end_tokens else []
    end_tok = [end_tokens[1]] if end_tokens else []
    # if len(text)==0:
    #     return torch.empty((1,0), dtype=torch.long)
    tok = torch.tensor([
        start_tok + [ord(char) for char in text] + end_tok
        ])
    rep = ' '+text+' ' if end_tokens else text
    idx_map = list(range(1,n+1) if end_tokens else range(n))
    return tok, rep, idx_map

class ZeroEncoder(nn.Module):
    r"""Just character embeddings.

    Args:
        in_out_channels (int): number of input and output channels.

    Shapes:
        - input: LongTensor (B, T)
        - output: (B, T, D)
    """
    def __init__(self, channels=768, end_tokens=True):
        super().__init__()
        self.end_tokens = [57344, 57345] if end_tokens else None
        # NOTE: stupid hack to support arbitrary number of embeddings
        # unicode points > 1024 may collide
        self.n_embeddings = 1025
        self.embed = nn.Embedding(self.n_embeddings, channels, padding_idx=0)
        
        self.channels = channels

    ### this function currently duplicated because torchscript is weird with inheritance
    @torch.jit.ignore
    def tokenize(self, text):
        return tokenize(text, self.end_tokens)

    @torch.jit.ignore
    def encode(self, x, mask=None):
        # if mask is not None:
            # raise NotImplementedError("please implement mask")
        # NOTE: stupid hack to support arbitrary number of embeddings
        return self.embed(x%self.n_embeddings)

class CanineEncoder(nn.Module):
    def __init__(self, 
            pretrained='google/canine-c', 
            end_tokens=True):
        # canine-c: pre-trained with autoregressive character loss
        # canine-s: pre-trained with subword masking loss
        super().__init__()
        self.end_tokens = [57344, 57345] if end_tokens else None
        # self.net = CanineModel.from_pretrained(pretrained)
        config = AutoConfig.from_pretrained(pretrained)
        self.net = AutoModel.from_config(config)

        self.channels = 768

    def init(self):
        """random initialization"""
        self.net = CanineModel(self.net.config)
    
    def pad(self, tokens, mask=None):
        pad = 4 - tokens.shape[1]
        if pad > 0:
            tokens = torch.cat((
                tokens, tokens.new_zeros(tokens.shape[0], pad)
                ), 1)
            if mask is not None:
                mask = torch.cat((
                    mask, mask.new_zeros(mask.shape[0], pad)
                ), 1)
        return tokens, mask

    def forward(self, text_t, mask):
        return self.net(text_t, mask, output_hidden_states=True)

    def encode(self, text, mask=None, layer=-1):
        if isinstance(text, str):
            text, _, _ = self.tokenize(text)
        n = text.shape[1]
        text, mask = self.pad(text, mask)
        h = self(text, mask).hidden_states[layer]
        h = h[:, :n] #unpad
        return h
    
    @torch.jit.ignore
    def tokenize(self, text):
        return tokenize(text, self.end_tokens)
    
class CanineEmbeddings(nn.Module):
    def __init__(self, 
            pretrained='google/canine-c', 
            end_tokens=True,
            bottleneck=False,
            use_positions=True
            ):
        # canine-c: pre-trained with autoregressive character loss
        # canine-s: pre-trained with subword masking loss
        super().__init__()
        self.end_tokens = [57344, 57345] if end_tokens else None
        self.net = CanineModel.from_pretrained(pretrained).char_embeddings
        if not use_positions:
            self.net.position_embedding_type = None
        self.channels = 768
        self.bottleneck = False
        if bottleneck:
            self.bottleneck = True
            self.proj = nn.Linear(self.channels, bottleneck)
            self.channels = bottleneck

    def init(self):
        """random initialization"""
        self.net = CanineModel(self.net.config)
    
    def pad(self, tokens, mask=None):
        pad = 4 - tokens.shape[1]
        if pad > 0:
            tokens = torch.cat((
                tokens, tokens.new_zeros(tokens.shape[0], pad)
                ), 1)
            if mask is not None:
                mask = torch.cat((
                    mask, mask.new_zeros(mask.shape[0], pad)
                ), 1)
        return tokens, mask

    def forward(self, text_t, mask):
        # no mixing, mask can be ignored
        h = self.net(text_t)
        if self.bottleneck:
            h = self.proj(h)
        return h

    def encode(self, text, mask=None):
        if isinstance(text, str):
            text, *_ = self.tokenize(text)
        # n = text.shape[1]
        # text, mask = self.pad(text, mask)
        h = self(text, mask)
        # h = h[:, :n] #unpad
        return h
    
    @torch.jit.ignore
    def tokenize(self, text):
        return tokenize(text, self.end_tokens)
    

# from coqui
class TacotronEncoder(nn.Module):
    r"""Tacotron2 style Encoder for comparison with CANINE.

    Args:
        in_out_channels (int): number of input and output channels.

    Shapes:
        - input: LongTensor (B, T)
        - output: (B, T, D)
    """
    def __init__(self, in_out_channels=768, end_tokens=True, conv_blocks=3, rnn=True):
        super().__init__()
        self.end_tokens = [57344, 57345] if end_tokens else None
        # NOTE: stupid hack to support arbitrary number of embeddings
        # unicode points > 1024 may collide
        self.n_embeddings = 1025
        self.embed = nn.Embedding(
            self.n_embeddings, in_out_channels, padding_idx=0)
        
        self.convolutions = nn.ModuleList()
        for _ in range(conv_blocks):
            self.convolutions.append(ConvBNBlock(
                in_out_channels, in_out_channels, 5, "relu"))
        if rnn:
            self.lstm = nn.LSTM(
                in_out_channels, int(in_out_channels / 2), num_layers=1, batch_first=True, bias=True, bidirectional=True
            )
        else:
            self.lstm = None
        # self.rnn_state = None
        self.channels = in_out_channels

    ### this function currently duplicated because torchscript is weird with inheritance
    @torch.jit.ignore
    def tokenize(self, text):
        return tokenize(text, self.end_tokens)

    @torch.jit.ignore
    def encode(self, x, mask=None):
        # if mask is not None:
            # raise NotImplementedError("please implement mask")
        input_lengths = (x>0).long().sum(-1)
        # NOTE: stupid hack to support arbitrary number of embeddings
        o = self.embed(x%self.n_embeddings).transpose(-1,-2)
        for layer in self.convolutions:
            o = o + layer(o, mask)
        o = o.transpose(-1, -2)
        if self.lstm is not None:
            o = nn.utils.rnn.pack_padded_sequence(
                o, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
            self.lstm.flatten_parameters()
            o, _ = self.lstm(o)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        return o

# from coqui
class ConvBNBlock(nn.Module):
    r"""Convolutions with Batch Normalization and non-linear activation.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): convolution kernel size.
        activation (str): 'relu', 'tanh', None (linear).

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_out, T)
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.convolution1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding)
        self.batch_normalization = nn.BatchNorm1d(
            out_channels, momentum=0.1, eps=1e-5)
        self.dropout = nn.Dropout(p=0.5)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x, mask:Optional[Tensor]=None):
        o = self.batch_normalization(x)
        o = self.activation(o)
        o = self.dropout(o)
        if mask is not None:
            # o = o.where(mask.bool()[:,None], 0)
            o = o.where(mask[:,None]>0, 0)
        o = self.convolution1d(o)
        return o
    
@ft.cache
def lev(t1, t2):
    """
    Returns:
        levenshtein distance between t1 and t2
        string representing edits from t1 to t2
            - delete
            + insert
            ^ edit
            . no change
        list giving the corresponding index in t2 for each position in t1
    """
    # print(t1, t2)
    if len(t1)==0:
        return len(t2), '+'*len(t2), []
    elif len(t2)==0:
        return len(t1), '-'*len(t1), [0]*len(t1)
    
    hd1, tl1 = t1[0], t1[1:]
    hd2, tl2 = t2[0], t2[1:]

    if hd1==hd2:
        n, s, i = lev(tl1, tl2)
        return n, '.'+s, [0]+[j+1 for j in i]
    (n, s, i), c, f = min(
        (lev(tl1, tl2), '^', lambda i: [0]+[j+1 for j in i]), 
        (lev(tl1, t2), '-', lambda i: [0]+i), 
        (lev(t1, tl2), '+', lambda i: [j+1 for j in i]))
    return n+1, c+s, f(i)