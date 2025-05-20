# adapted from coqui TTS
# Mozilla Public License

from typing import Dict, Optional, Tuple
import math
import itertools as it

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import normflows as nf

from normflows.distributions.base import ConditionalDiagGaussian

import numpy as np

from tungnaa.text import CanineEncoder, CanineEmbeddings, TacotronEncoder, ZeroEncoder

def hz_to_z(hz):
    return torch.where(hz>0, hz.log2() - 7., torch.zeros_like(hz))
def z_to_hz(z):
    return 2**(z+7.)


def categorical_sample(logits, n:int):
    return torch.multinomial(logits.exp()+1e-10, n, replacement=True)

def categorical_sample_2d(logits, n:int):

    return torch.multinomial(
        logits.reshape(-1, logits.shape[2]).exp().clamp(1e-10, 1e10), n, replacement=True
        ).reshape(logits.shape[0], logits.shape[1], n)

# from scipy.stats import betabinom

# this implements the beta-binomial distribution PMF
# to eliminate the scipy dependency
def prior_coefs(prior_filter_len, alpha, beta):
    lg = torch.special.gammaln
    n = torch.tensor(prior_filter_len-1)
    k = torch.arange(prior_filter_len)
    a = torch.tensor(alpha)
    b = torch.tensor(beta)
    log_prior = (
        lg(n+1) - lg(k+1) - lg(n-k+1)
        + lg(k+a) + lg(n-k+b) - lg(n+a+b) 
        - lg(a) - lg(b) + lg(a+b)
    )
    prior = log_prior.exp()
    return prior.float().flip(0)[None,None]

# @torch.jit.script
def do_prior_filter(attn_weight, filt, pad:int):
    prior_filter = F.conv1d(
        F.pad(attn_weight.unsqueeze(1), (pad, 0)), filt)
    return prior_filter.clamp_min(1e-6).log().squeeze(1)

# @torch.jit.script
def addsoftmax(a, b):
    return (a + b).softmax(-1)

class GaussianAttention(nn.Module):
    """
    More restrictive version of dynamic convolutional attention.
    the convolution is always a difference of gaussians (diffusion / unsharp mask)
    with a nonnegative shift.

    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the value tensor.
    """

    def __init__(
        self,
        query_dim,
        filter_size:int=21,
        min_sigma:float=0.2,
        prior_tokens_per_frame:float=1.0,
        wn=False,
    ):
        super().__init__()
        self._mask_value = 1e-8

        norm = nn.utils.parametrizations.weight_norm if wn else lambda x:x
        self.proj = norm(nn.Linear(query_dim, 4))
        self.filter_size = filter_size
        self.min_sigma = min_sigma
        self.prior_tokens_per_frame = prior_tokens_per_frame

        self.register_buffer(
            'k', torch.arange(filter_size)[:, None, None], persistent=False)

    def align(self, 
            query, attention_weights, 
            inputs:Optional[Tensor]=None, mask:Optional[Tensor]=None):
        """
        Args:
            query: [B, D_attn_rnn]
            attention_weights: [B, T_text]
            inputs: [B, T_text, D_text]
            mask: [B, T_text]
        Returns:
            attention_weights: [B, T_text]
        """
        B = query.size(0)

        mu, alpha, sigma = F.sigmoid(self.proj(query)).split((1,1,2),dim=1)
        # mu, alpha, sigma = F.softplus(self.proj(query)).split((1,1,2),dim=1)
        # mu = self.filter_size//2 - mu * self.prior_tokens_per_frame# shift
        mu = self.filter_size//2 - mu * (2 * self.prior_tokens_per_frame)# shift
        # alpha, beta = 1+alpha, -alpha # diffuse weight, unsharp weight
        sigma = self.min_sigma + sigma.cumsum(-1) # second more dispersed than first

        # NOTE: may be able to compute k using torch.special.ndtr + diff?

        k = self.k
        # k = torch.arange(
        #     self.filter_size, 
        #     device=query.device, dtype=query.dtype
        #     )[:, None, None] # [K, 1, 1]

        # g = (-(k-mu)**2/(2*sigma**2)).exp()/sigma
        g = (-((k-mu)/(2*sigma))**2).exp()/sigma

        alpha = alpha.T
        k = (1+alpha)*g[...,0] - alpha*g[...,1] # [K, B]
        k = k/k.sum(0) # normalize
        k = k.T # [B, K]

        attention_weights = F.conv1d(
            attention_weights[None], k[:,None], 
            groups=B, padding=self.filter_size//2
        ).squeeze(0)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~mask, self._mask_value)

        attention_weights = attention_weights.clip(self._mask_value)

        attention_weights = (
            attention_weights / attention_weights.sum(-1, keepdim=True))
        
        return attention_weights

    def apply(self, attention_weights, inputs, mask:Optional[Tensor]=None):
        """this is split out to implement attention painting
        """
        # do weights really need to be masked here if it's post-softmax anyway?
        # is it enough that the inputs are zero padded?
        # print(attention_weights.shape, inputs.shape)
        # apply masking
        # if mask is not None:
        #     if torch.is_grad_enabled():
        #         attention_weights = attention_weights.masked_fill(
        #             ~mask, self._mask_value)
        #     else:
        #         attention_weights.masked_fill_(~mask, self._mask_value)
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return context

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        attention_weights = inputs.new_zeros(B, T)
        attention_weights[:, 0] = 1.0
        return attention_weights
    

class MonotonicDynamicConvolutionAttention(nn.Module):
    """Dynamic convolution attention from
    https://arxiv.org/pdf/1910.10288.pdf

    This is an altered version where the static filter is replaced by the bias 
    of the linear layer leading to the dynamic filter

    original docstring follows:

    query -> linear -> tanh -> linear ->|
                                        |                                            mask values
                                        v                                              |    |
               atten_w(t-1) -|-> conv1d_dynamic -> linear -|-> tanh -> + -> softmax -> * -> * -> context
                             |-> conv1d_static  -> linear -|           |
                             |-> conv1d_prior   -> log ----------------|
    query: attention rnn output.
    Note:
        Dynamic convolution attention is an alternation of the location senstive attention with
    dynamically computed convolution filters from the previous attention scores and a set of
    constraints to keep the attention alignment diagonal.
        DCA is sensitive to mixed precision training and might cause instable training.
    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the value tensor.
        static_filter_dim (int): number of channels in the convolution layer computing the static filters.
        static_kernel_size (int): kernel size for the convolution layer computing the static filters.
        dynamic_filter_dim (int): number of channels in the convolution layer computing the dynamic filters.
        dynamic_kernel_size (int): kernel size for the convolution layer computing the dynamic filters.
        prior_filter_len (int, optional): [description]. Defaults to 11 from the paper.
        alpha (float, optional): [description]. Defaults to 0.1 from the paper.
        beta (float, optional): [description]. Defaults to 0.9 from the paper.
    """

    def __init__(
        self,
        query_dim,
        attention_dim,
        static_filter_dim=8, # unused
        static_kernel_size=21, # unused
        dynamic_filter_dim=8,
        dynamic_kernel_size=21,
        prior_filter_len=11,
        alpha=0.1,
        beta=0.9,
    ):
        super().__init__()
        self._mask_value = 1e-8
        self.dynamic_filter_dim = dynamic_filter_dim
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_filter_len = prior_filter_len
        # setup key and query layers
        dynamic_weight_dim = dynamic_filter_dim * dynamic_kernel_size
        # self.filter_mlp = torch.jit.script(nn.Sequential(
        self.filter_mlp = (nn.Sequential(
            nn.Linear(query_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, dynamic_weight_dim)#, bias=False)
        ))

        # self.post_mlp = torch.jit.script(nn.Sequential(
        self.post_mlp = (nn.Sequential(
            nn.Linear(dynamic_filter_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        ))

        self.register_buffer(
            "prior", prior_coefs(prior_filter_len, alpha, beta), 
            persistent=False)

    def align(self, 
            query, attention_weights, 
            inputs:Optional[Tensor]=None, mask:Optional[Tensor]=None):
        """
        Args:
            query: [B, D_attn_rnn]
            attention_weights: [B, T_text]
            inputs: [B, T_text, D_text]
            mask: [B, T_text]
        Returns:
            attention_weights: [B, T_text]
        """
        B = query.shape[0]

        prior_filter = do_prior_filter(
            attention_weights, self.prior, self.prior_filter_len-1)

        G = self.filter_mlp(query)
        # compute dynamic filters
        pad = (self.dynamic_kernel_size - 1) // 2
        dynamic_filter = F.conv1d(
            attention_weights[None],
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=pad,
            groups=B,
        )
        dynamic_filter = dynamic_filter.view(
            B, self.dynamic_filter_dim, -1).transpose(1, 2)

        attention_weights = addsoftmax(
            self.post_mlp(dynamic_filter).squeeze(-1), prior_filter)
        
        return attention_weights

    def apply(self, attention_weights, inputs, mask:Optional[Tensor]=None):
        """this is split out to implement attention painting

        do weights really need to be masked here if it's post-softmax anyway?
        is it enough that the inputs are zero padded?
        """
        # print(attention_weights.shape, inputs.shape)
        # apply masking
        if mask is not None:
            if torch.is_grad_enabled():
                attention_weights = attention_weights.masked_fill(
                    ~mask, self._mask_value)
            else:
                attention_weights.masked_fill_(~mask, self._mask_value)
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return context

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        attention_weights = inputs.new_zeros(B, T)
        attention_weights[:, 0] = 1.0
        return attention_weights


# @torch.jit.script
def zoneout(x1:Tensor, x2:Tensor, p:float, training:bool):
    """
    Args:
        x1: old value
        x2: new value
        p: prob of keeping old value
        training: stochastic if True, expectation if False
    """
    keep = torch.full_like(x1, p)
    if training:
        keep = torch.bernoulli(keep)
    return torch.lerp(x2, x1, keep)

class ResidualRNN(nn.Module):
    def __init__(self, cls:type, input_size, hidden_size, 
            dropout=0, layers=1, **kw):
        super().__init__()
        self.dropout = dropout
        net = []
        for _ in range(layers):
            net.append(cls(input_size,hidden_size,**kw))
            input_size = hidden_size
        self.net = nn.ModuleList(net)

    def forward(self, 
            x, states:Tuple[Tensor, Tensor], 
            training:bool, 
            dropout_p:float=0.1, dropout_type:Optional[str]=None):
        hs, cs = states
        hs_out, cs_out = [], []
        # for i,layer,h,c in zip(range(len(self.net)), self.net, hs, cs):
        for i,layer in enumerate(self.net):
            h,c = hs[i], cs[i]
            h,c = layer(
                F.dropout(x, self.dropout, self.training), (h,c), training, dropout_p, dropout_type)
            x = x + h if i > 0 else h
            hs_out.append(h)
            cs_out.append(c)

        hs_out = torch.stack(hs_out)
        cs_out = torch.stack(cs_out)

        return hs_out, cs_out

class DropoutRNN(nn.Module):
    def __init__(self, rnn:nn.Module):
        super().__init__()
        self.rnn = rnn

    def forward(self, x, states:Tuple[Tensor, Tensor], 
        training:bool, dropout_p:float=0.1, dropout_type:Optional[str]=None):
        h, cell = self.rnn(x, states)
        if dropout_type is None:
            new_states = h, cell
        elif dropout_type=='dropout':
            new_states = (
                F.dropout(h, dropout_p, training=training),
                F.dropout(cell, dropout_p, training=training) 
            )
            # new_states = (
            #     F.dropout(s, dropout_p, training=training) 
            #     for s in new_states)
        elif dropout_type=='zoneout':
            new_states = (
                zoneout(states[0], h, dropout_p, training=training), 
                zoneout(states[1], cell, dropout_p, training=training) 
            )
        else: raise ValueError(dropout_type)
        return new_states


class DiagonalNormalMixture(nn.Module):
    def __init__(self, n:int=16):
        """n: number of mixture components"""
        super().__init__()
        self.n = n
        self.register_buffer(
            'log2pi', torch.tensor(np.log(2*np.pi)).float(), persistent=False)

    def n_params(self, size):
        """# of distribution parameters as a function of # latent variables"""
        return (2 * size + 1) * self.n

    def get_params(self, params:Tensor):
        """
        Args:
            params: Tensor[batch, time, n_params] 
        Returns:
            mu: Tensor[batch, time, n_latent, self.n]
            logsigma: Tensor[batch, time, n_latent, self.n]
            logitpi: Tensor[batch, time, self.n]
        """
        #means, log stddevs, logit weights
        locscale = params[...,:-self.n]
        logitpi = params[...,-self.n:]

        mu, logsigma = locscale.view(
            params.shape[0], params.shape[1], -1, self.n, 2).unbind(-1)
        # mu, logsigma = locscale.view(*params.shape[:-1], -1, self.n, 2).unbind(-1)
        return mu, logsigma, logitpi

    def forward(self, x:Tensor, params:Tensor, mode:str='normal'):
        """mixture of diagonal normals negative log likelihood.
        should broadcast any number of leading dimensions
        Args:
            x: Tensor[batch, time, latent] or [..., latent]
            params: Tensor[batch, time, n_params] or [..., n_params]
        Return:
            negative log likelihood: Tensor[batch, time] or [...]
        """
        x = x[...,None] # broadcast against mixture component
        mu, logsigma, logitpi = self.get_params(params)

        if mode == 'fixed':
            logsigma = torch.zeros_like(logsigma)
        elif mode == 'posthoc':
            logsigma = (x - mu).abs().log()
        elif mode != 'normal':
            raise ValueError(mode)
    
        logsigma = logsigma.clip(-7, 5) # TODO: clip range?
        # cmp_loglik = -0.5 * (
        #     ((x - mu) / logsigma.exp()) ** 2
        #     + log2pi
        # ) - logsigma
        # logpi = logitpi.log_softmax(-1)
        # return -(cmp_loglik.sum(-2) + logpi).logsumexp(-1)
        cmp_loglik = (
            (x - mu) # materialize huge tensor when calling from sample_n
            / logsigma.exp()) ** 2 

        # reduce latent dim
        # reduce before summing these components for memory reasons
        cmp_loglik = -0.5 * (
            (cmp_loglik.sum(-2) + self.log2pi*x.shape[-2]) 
            ) - logsigma.sum(-2)

        logpi = logitpi.log_softmax(-1)
        return -(cmp_loglik + logpi).logsumexp(-1)
    
    def sample(self, params:Tensor, temperature:float=1.0, nsamp:Optional[int]=None):
        # print(params.shape, temperature, nsamp)
        if temperature != 1:
            # return self.sample_n(params, temperature, nsamp=128)
            if nsamp is not None:
                return self.sample_n(params, temperature, nsamp=nsamp)
            else:
                return self.sample_components(params, temperature)
        else:
            mu, logsigma, logitpi = self.get_params(params)
            # idx = torch.distributions.Categorical(
                # logits=logitpi).sample()[...,None,None].expand(*mu.shape[:-1],1)
            idx = categorical_sample_2d(logitpi, 1)[...,None].expand(
                    mu.shape[0], mu.shape[1], mu.shape[2], 1)
            mu = mu.gather(-1, idx).squeeze(-1)
            logsigma = logsigma.gather(-1, idx).squeeze(-1)
            return mu + logsigma.exp()*torch.randn_like(logsigma)
            # return mu + temperature*logsigma.exp()*torch.randn_like(logsigma)
    
    def sample_n(self, params:Tensor, temperature:float=1.0, nsamp:int=128):
        """
        draw nsamp samples,
        rerank and sample categorical with temperature
        Args:
            params: Tensor[batch, time, n_params]    
        """
         # sample N 
        mu, logsigma, logitpi = self.get_params(params)
        # logitpi = logitpi[...,None,:].expand(*logitpi.shape[:-1],nsamp,-1)
        # logitpi = logitpi[...,None,:].expand(
            # logitpi.shape[0],logitpi.shape[1],nsamp,-1)
        # ..., nsamp, self.n
        # print(f'{logitpi.shape=}')
        # idx = torch.distributions.Categorical(logits=logitpi).sample()
        # idx = torch.multinomial(
            # logitpi.reshape(-1, logitpi.shape[-1]).exp(), nsamp, replacement=True
            # ).reshape(logitpi.shape[0], logitpi.shape[1], nsamp)
        idx = categorical_sample_2d(logitpi, nsamp)
        # ..., nsamp
        # print(f'{idx.shape=}')
        # idx = idx[...,None,:].expand(*mu.shape[:-1], -1)
        idx = idx[...,None,:].expand(mu.shape[0], mu.shape[1], mu.shape[2], -1)
        # ..., latent, nsamp 
        # print(f'{idx.shape=}')

        # mu is: ..., latent, self.n
        mu = mu.gather(-1, idx)
        logsigma = logsigma.gather(-1, idx)
        # ..., latent, nsamp
        # print(f'{mu.shape=}')

        samps = (mu + torch.randn_like(mu)*logsigma.exp()).moveaxis(-1, 0)
        # nsamp,...,latent
        # print(f'{samps.shape=}')

        # compute nll
        # here there is a extra first dimension (nsamp)
        # which broadcasts against the distribution params inside of self.forward,
        # to compute the nll for each sample
        nll = self(samps, params).moveaxis(0, -1)
        # ...,nsamp
        # print(f'{nll.shape=}')
        # print(f'{nll=}')

        # sample categorical with temperature
        # idx = torch.distributions.Categorical(
        #     logits=-nll/(temperature+1e-5)).sample()
        # ...
        idx = categorical_sample_2d(-nll/(temperature+1e-5), 1)
        # print(f'{idx.shape=}')
        # ...,1

        # select
        # idx = idx[None,...,None].expand(1, *samps.shape[1:])
        # 1,...,latent
        idx = idx[None,...].expand(1, samps.shape[1], samps.shape[2], samps.shape[3])
        # 1,...,latent
        # print(f'{idx.shape=}')
        samp = samps.gather(0, idx).squeeze(0)
        # ...,latent
        # print(f'{samp.shape=}')
        # print(f'{samp=}')
        return samp

    def sample_components(self, params:Tensor, temperature:float=1.0):
        """
        sample every mixture component with temperature,
        rerank and sample categorical with temperature.
        """
        # sample each component with temperature 
        mu, logsigma, _ = self.get_params(params)
        samps = mu + torch.randn_like(mu)*logsigma.exp()*temperature**0.5
        # ..., latent, self.n
        samps = samps.moveaxis(-1, 0)
        # self.n ...latent

        # compute nll for each sample
        # here there is an extra first dimension (nsamp)
        # which broadcasts against the distribution params inside of self.forward
        # to compute the nll for each sample
        nll = self(samps, params).moveaxis(0, -1)
        # ..., self.n

        # sample categorical with temperature
        if temperature > 1e-5:
            logits = nll.mul_(-1/temperature**0.5)
            # idx = torch.distributions.Categorical(logits=logits).sample()
            idx = categorical_sample_2d(logits, 1)
        else:
            idx = nll.argmin(-1)[...,None]
        # print(f'{idx.shape=}')
        # ...

        # select
        # idx = idx[None,...,None].expand(1, *samps.shape[1:])
        idx = idx[None,...].expand(1, samps.shape[1], samps.shape[2], samps.shape[3])

        # 1,...,latent
        # print(f'{idx.shape=}')
        samp = samps.gather(0, idx).squeeze(0)
        # ...,latent
        # print(f'{samp.shape=}')
        # print(samp.shape)
        # print(f'{samp=}')
        return samp

    @torch.jit.ignore
    def metrics(self, params:Tensor):
        mu, logsigma, logitpi = self.get_params(params)
        ent = torch.distributions.Categorical(logits=logitpi).entropy().detach()
        return {
            'logsigma_min': logsigma.min().detach(),
            'logsigma_median': logsigma.median().detach(),
            'logsigma_max': logsigma.max().detach(),
            'pi_entropy_min': ent.min(),
            'pi_entropy_median': ent.median(),
            'pi_entropy_max': ent.max(),
        } 

class NSF(nn.Module):
    """Neural Spline Flow likelihood"""
    def __init__(self, 
            latent_size,
            context_size=256,
            hidden_size=256,
            hidden_layers=2,
            blocks=8,
            bins=8,
            dropout=None,
            ):
        super().__init__()
        self.context_size = context_size
        self.latent_size = latent_size

        flows = []
        for _ in range(blocks):
            flows.append(nf.flows.CoupledRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_size, context_size,
                num_bins=bins, dropout_probability=dropout))
            flows.append(nf.flows.LULinearPermute(latent_size))
        # base distribution
        # q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
        q0 = ConditionalDiagGaussian(
            latent_size, context_encoder=nn.Linear(context_size, 2*latent_size))
        # flow model
        self.net = nf.ConditionalNormalizingFlow(q0=q0, flows=flows)
        # self.net.q0.forward = torch.jit.ignore(self.net.q0.forward)

    def n_params(self, size):
        assert size == self.latent_size
        return self.context_size

    @torch.jit.ignore
    def forward(self, x:Tensor, params:Tensor):
        """negative log likelihood
        Args:
            x: Tensor[batch, time, channel]
            params: Tensor[batch, time, n_params]    
        Return:
            negative log likelihood: Tensor[batch, time]
        """
        bs = torch.broadcast_shapes(x.shape[:-1], params.shape[:-1])
        params = params.expand(*bs, -1).reshape(-1, params.shape[-1])
        x = x.expand(*bs, -1).reshape(-1, x.shape[-1])
        loglik = self.net.log_prob(x, context=params)
        return -loglik.reshape(*bs)
    
    @torch.jit.ignore
    def sample(self, params:Tensor, temperature:float=1.0):
        bs = params.shape[:-1]
        params = params.reshape(-1, params.shape[-1])
        # if temperature!=1:
        #     print('warning: temperature not implemented in NSF')
        # samp, logprob = self.net.sample(params.shape[0], context=params)
        mu, logsigma = self.net.q0.context_encoder(params).chunk(2,-1)
        base_samp = mu + torch.randn_like(mu)*logsigma.exp()*temperature
        samp = self.net(base_samp, context=params)
        samp = samp.reshape(*bs, -1)
        return samp
    
    def metrics(self, params:Tensor):
        return {}
    
class GED(nn.Module):
    """Generalized energy distance (pseudo) likelihood"""
    def __init__(self, 
            latent_size,
            hidden_size=256,
            hidden_layers=3,
            dropout=None,
            unfold=None,
            multiply_params=False,
            project_params=None,
            glu=False,
            ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.unfold = unfold
        self.multiply_params = multiply_params
        self.project_params = project_params

        layers = []
        norm = nn.utils.parametrizations.weight_norm 
        if glu:
            act = nn.GLU
            for _ in range(hidden_layers):
                block = [norm(nn.Linear(hidden_size, 2*hidden_size))]
                block.append(act())
                if dropout:
                    block.append(nn.Dropout(dropout))
                layers.append(Residual(*block))
            layers.append(norm(nn.Linear(hidden_size, latent_size)))
        else:
            act = nn.GELU
            for _ in range(hidden_layers):
                block = [nn.Dropout(dropout)] if dropout else []
                block.append(act())
                block.append(norm(nn.Linear(hidden_size, hidden_size)))
                layers.append(Residual(*block))
            layers.extend([
                act(),
                norm(nn.Linear(hidden_size, latent_size))
                ])

        self.net = nn.Sequential(*layers)

    def n_params(self, size):
        assert size == self.latent_size
        if self.project_params:
            p = self.project_params
            return self.hidden_size + p*(p-1)
        elif self.multiply_params:
            return self.hidden_size
        else:
            return self.hidden_size//2

    @torch.jit.ignore
    def forward(self, x:Tensor, params:Tensor):
        """compute generalized energy distance
        Args:
            x: Tensor[batch, time, channel]
            params: Tensor[batch, time, n_params]
        Return:
            negative log likelihood: Tensor[batch, time]
        """
        # bs = torch.broadcast_shapes(x.shape[:-1], params.shape[:-1])
        # params = params.expand(*bs, -1).reshape(-1, params.shape[-1])
        # x = x.expand(*bs, -1).reshape(-1, x.shape[-1])
        params = torch.cat((params, params), 0)
        y = self.net(self.add_noise(params))
        if self.unfold:
            zeros = y.new_zeros(*y.shape[:-2],self.unfold-1,y.shape[-1])
            y = torch.cat((zeros, y), -2)
            y = y.unfold(dimension=-2, size=self.unfold, step=1)
            zeros = x.new_zeros(*x.shape[:-2],self.unfold-1,x.shape[-1])
            x = torch.cat((zeros, x), -2)
            x = x.unfold(dimension=-2, size=self.unfold, step=1)
            dim = (-1,-2)
        else:
            dim = -1
        y1, y2 = y.chunk(2, 0)
        return (
            torch.linalg.vector_norm(x-y1, dim=dim)
            + torch.linalg.vector_norm(x-y2, dim=dim)
            - torch.linalg.vector_norm(y2-y1, dim=dim)
            )
                
    # @torch.jit.ignore
    def sample(self, params, temperature:float=1.0):
        return self.net(self.add_noise(params, temperature))
    
    def add_noise(self, params, temperature:float=1):
        if self.project_params is not None:
            p = self.project_params
            w, params = params.split((p**2),-1)
            noise = torch.randn(
                *params.shape[:2],1,p, device=params.device, dtype=params.dtype)
            w = w.reshape(*w.shape[:2],p,p)
            noise = (noise @ w).squeeze(-2)
        else:
            if self.multiply_params:
                params, w = params.chunk(2,-1)
                noise = w*torch.randn_like(params)*temperature
            else:
                noise = torch.randn_like(params)*temperature

        return torch.cat((params, noise), -1)
    
    def metrics(self, params:Tensor):
        return {}
    
class StandardNormal(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'log2pi', torch.tensor(np.log(2*np.pi)).float(), persistent=False)

    def n_params(self, size):
        return size

    def forward(self, x:Tensor, params:Tensor):
        """standard normal negative log likelihood
        Args:
            x: Tensor[batch, time, channel]
            params: Tensor[batch, time, n_params]    
        Return:
            negative log likelihood: Tensor[batch, time]
        """
        mu = params
        loglik = 0.5 * (
            (x - mu) ** 2 + self.log2pi
        )
        return loglik.sum(-1)
    
    def sample(self, params:Tensor, temperature:float=1.0):
        return params + temperature*torch.randn_like(params)
    
    def metrics(self, params:Tensor):
        return {}

class DiagonalNormal(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'log2pi', torch.tensor(np.log(2*np.pi)).float(), persistent=False)

    def n_params(self, size):
        return size * 2

    def forward(self, x:Tensor, params:Tensor):
        """diagonal normal negative log likelihood
        Args:
            x: Tensor[batch, time, channel]
            params: Tensor[batch, time, n_params]    
        Return:
            negative log likelihood: Tensor[batch, time]
        """
        mu, logsigma = params.chunk(2, -1)
        logsigma = logsigma.clip(-7, 5) # TODO: clip range
        loglik = 0.5 * (
            ((x - mu) / logsigma.exp()) ** 2
            + self.log2pi
        ) + logsigma
        return loglik.sum(-1)
    
    def sample(self, params:Tensor, temperature:float=1.0):
        mu, logsigma = params.chunk(2, -1)
        return mu + temperature*logsigma.exp()*torch.randn_like(logsigma)
    
    def metrics(self, params:Tensor):
        mu, logsigma = params.chunk(2, -1)
        return {
            'logsigma_min': logsigma.min().detach(),
            'logsigma_median': logsigma.median().detach(),
            'logsigma_max': logsigma.max().detach(),
        }

# adapted from https://github.com/NVIDIA/tacotron2/
class TacotronDecoder(nn.Module):
    def __init__(
        self,
        in_channels=None, # text embedding dim
        frame_channels=None, # RAVE latent dim
        dropout=0.1,
        likelihood_type='nsf',#'normal'#'mixture'#'ged'
        mixture_n=16,
        flow_context=256,
        flow_hidden=256,
        flow_layers=2,
        flow_blocks=2,
        nsf_bins=16,
        ged_hidden=256,
        ged_layers=4,
        ged_unfold=None,
        ged_multiply_params=False,
        ged_project_params=None,
        ged_glu=False,
        ged_dropout=None, #None follows main dropout, 0 turns off
        dropout_type='dropout', #'zoneout'
        prenet_type='original', # disabled
        prenet_dropout=0.2,
        prenet_layers=2,
        prenet_size=256,
        prenet_wn=False,
        hidden_dropout=0,
        separate_stopnet=True, # disabled
        max_decoder_steps=10000,
        text_encoder:Dict=None,
        rnn_size=1200,
        rnn_bias=True,
        rnn_layers=1,
        noise_channels=0,
        decoder_type='lstm',
        decoder_layers=1,
        decoder_size=None,
        hidden_to_decoder=True,
        memory_to_decoder=False,
        init_proj=1.0,
        proj_wn=False,
        attn_wn=False,
        learn_go_frame=False,
        pitch_xform=False,
        length_reparam=False,
        text_encoder_type='canine',
        block_size=2048,
        max_batch=8,
        max_tokens=1024,
        prior_filter_len=11,
        tokens_per_frame=1.0,
        attention_type='dca',
        script=False
    ):
        """
        Args:
            in_channels (int): number of input channels.
            frame_channels (int): number of feature frame channels.
            dropout (float): dropout rate (except prenet).
            prenet_dropout (float): prenet dropout rate.
            max_decoder_steps (int): Maximum number of steps allowed for the decoder. Defaults to 10000.
            text_encoder: dict of text encoder kwargs
        """
        super().__init__()
        assert frame_channels is not None

        if length_reparam:
            frame_channels = frame_channels + 1

        self.B = max_batch
        self.T = max_tokens

        if text_encoder_type not in [None, 'none']:
            if text_encoder is None: text_encoder = {}
            if text_encoder_type=='zero':
                self.text_encoder = ZeroEncoder(**text_encoder)
            elif text_encoder_type=='baseline':
                self.text_encoder = TacotronEncoder(**text_encoder)
            elif text_encoder_type=='canine':
                self.text_encoder = CanineEncoder(**text_encoder)
            elif text_encoder_type=='canine_embedding':
                self.text_encoder = CanineEmbeddings(**text_encoder)
            else:
                raise ValueError(text_encoder_type)
            if in_channels is None:
                in_channels = self.text_encoder.channels
            elif in_channels != self.text_encoder.channels:
                raise ValueError(f'{in_channels=} but {self.text_encoder.channels=}')
        else:
            self.text_encoder = None
            assert in_channels is not None

        self.max_decoder_steps = max_decoder_steps
        self.block_size = block_size
        self.frame_channels = frame_channels
        # self.pitch_xform = pitch_xform
        # # self.r_init = r
        # # self.r = r
        # self.encoder_embedding_dim = in_channels
        # self.separate_stopnet = separate_stopnet
        # self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = 0.5

        # decoder_size = decoder_size or rnn_size

        # # model dimensions
        # self.decoder_layers = decoder_layers
        # self.attention_hidden_dim = rnn_size
        # self.decoder_rnn_dim = decoder_size
        # self.prenet_dim = prenet_size
        # # self.attn_dim = 128
        # self.p_attention_dropout = dropout
        # self.p_decoder_dropout = dropout
        # self.dropout_type = dropout_type

        self.prenet_dropout = prenet_dropout
      
        # the likelihood converts a hidden state to a probability distribution
        # over each vocoder frame.
        # in the simpler cases this is just unpacking location and scale from
        # the hidden state.
        # in other cases the likelihood can be a normalizing flow with its own
        # trainable parameters.
        if likelihood_type=='normal':
            self.likelihood = StandardNormal()
        elif likelihood_type=='diagonal':
            self.likelihood = DiagonalNormal()
        elif likelihood_type=='mixture':
            self.likelihood = DiagonalNormalMixture(mixture_n)
        elif likelihood_type=='ged':
            ged_dropout = dropout if ged_dropout is None else ged_dropout
            self.likelihood = GED(
                self.frame_channels,
                hidden_size=ged_hidden, hidden_layers=ged_layers, 
                dropout=ged_dropout, unfold=ged_unfold, 
                multiply_params=ged_multiply_params,
                project_params=ged_project_params,
                glu=ged_glu
                )
        elif likelihood_type=='nsf':
            self.likelihood = NSF(
                self.frame_channels, context_size=flow_context,
                hidden_size=flow_hidden, hidden_layers=flow_layers,
                blocks=flow_blocks, bins=nsf_bins,
                dropout=dropout)
        else:
            raise ValueError(likelihood_type)
        
        if script and likelihood_type!='nsf':
            self.likelihood = torch.jit.script(self.likelihood)

        self.core = TacotronCore(
            in_channels=in_channels, # text embedding dim
            out_channels=self.likelihood.n_params(self.frame_channels),
            frame_channels=frame_channels, # RAVE latent dim
            dropout=dropout,
            dropout_type=dropout_type, #'zoneout'
            prenet_type=prenet_type, # disabled
            prenet_dropout=prenet_dropout,
            prenet_layers=prenet_layers,
            prenet_size=prenet_size,
            prenet_wn=prenet_wn,
            hidden_dropout=hidden_dropout,
            separate_stopnet=separate_stopnet, # disabled
            rnn_size=rnn_size,
            rnn_bias=rnn_bias,
            rnn_layers=rnn_layers,
            noise_channels=noise_channels,
            decoder_type=decoder_type,
            decoder_layers=decoder_layers,
            decoder_size=decoder_size,
            hidden_to_decoder=hidden_to_decoder,
            memory_to_decoder=memory_to_decoder,
            init_proj=init_proj,
            proj_wn=proj_wn,
            attn_wn=attn_wn,
            learn_go_frame=learn_go_frame,
            pitch_xform=pitch_xform,
            block_size=block_size,
            max_batch=max_batch,
            max_tokens=max_tokens,
            prior_filter_len=prior_filter_len,
            tokens_per_frame=tokens_per_frame,
            attention_type=attention_type,
            length_reparam=length_reparam
        )

        if script:
            self.core = torch.jit.script(self.core)

    @property
    def memory(self):
        return self.core.memory
    
    @classmethod
    def from_checkpoint(cls, path_or_dict):
        if isinstance(path_or_dict, dict):
            ckpt = path_or_dict
        else:
            ckpt = torch.load(
                path_or_dict, map_location='cpu', weights_only=False)

        kw = ckpt['kw']
        model_kw = cls.update_kw_dict(kw['model'])

        model = cls(**model_kw)
        try:
            model.load_state_dict(ckpt['model_state'], strict=True)
        except Exception as e:
            print(e.__traceback__)
            model.load_state_dict(ckpt['model_state'], strict=False)
    

        return model

    @classmethod
    def update_kw_dict(cls, d):
        """backward compatibility with older checkpoints"""
        b = d.pop('text_bottleneck', None)
        if b is not None:
            if d['text_encoder'] is None:
                d['text_encoder'] = {}
            d['text_encoder']['bottleneck'] = b
        return d

    @torch.jit.ignore
    def update_state_dict(self, d):
        """backward compatibility with older checkpoints"""
        # TODO: core.
        def replace(old, new):
            t = d.pop(old, None)
            if t is not None:
                d[new] = t
        replace('go_attention_rnn_cell_state', 'go_attention_cell')
        replace('go_query', 'go_attention_hidden')
        # rnncell -> dropoutrnn
        replace('attention_rnn.weight_hh', 'attention_rnn.rnn.weight_hh')
        replace('attention_rnn.weight_ih', 'attention_rnn.rnn.weight_ih')
        replace('attention_rnn.bias_hh', 'attention_rnn.rnn.bias_hh')
        replace('attention_rnn.bias_ih', 'attention_rnn.rnn.bias_ih')
        # dropoutrnn -> residualrnn
        replace('attention_rnn.rnn.weight_hh', 'core.attention_rnn.net.0.rnn.weight_hh')
        replace('attention_rnn.rnn.weight_ih', 'core.attention_rnn.net.0.rnn.weight_ih')
        replace('attention_rnn.rnn.bias_hh', 'core.attention_rnn.net.0.rnn.bias_hh')
        replace('attention_rnn.rnn.bias_ih','core.attention_rnn.net.0.rnn.bias_ih')
        for name in (
            'go_attention_hidden', 'go_attention_cell',
            'attention_hidden', 'attention_cell'
            ):
            if name in d and d[name].ndim==2:
                d[name] = d[name][None]
        # move into core
        for name in list(d):
            if any(name.startswith(s) for s in (
                "go_", "memory", "context", "attention_", "decoder_", "alignment", "inputs", "prenet.", "attention.", "decoder_rnn.", "linear_projection.", "stopnet.")):
                replace(name, f'core.{name}')

        # text bottleneck -> text encoder
        replace('core.text_proj.weight', 'text_encoder.proj.weight')
        replace('core.text_proj.bias', 'text_encoder.proj.bias')
        return d

    @torch.jit.ignore
    def load_state_dict(self, d, **kw):
        super().load_state_dict(self.update_state_dict(d), **kw)
         
    @torch.jit.export
    def reset(self, inputs):
        r"""
        populates buffers with initial states and text inputs

        call with encoded text, before using `step`

        Args:
            inputs: (B, T_text, D_text)

        """
        self.core.reset(inputs)

    # TODO: should there be option to include acoustic memory here?
    @torch.jit.export
    def get_state(self):
        return self.core.get_state()
    
    @torch.jit.export
    def set_state(self, 
            state:Dict[str,Tensor|Tuple[Tensor, Tensor, Tensor, Tensor]]):
        self.core.set_state(state)

    def latent_map(self, z):
        return self.core.latent_map(z)

    def latent_unmap(self, z):
        return self.core.latent_unmap(z)
    
    def chunk_pad(self, inputs, mask, c=128):
        b, t = mask.shape
        p = math.ceil(t / c) * c - t
        if p>0:
            inputs = torch.cat(
                (inputs, inputs.new_zeros(b, p, *inputs.shape[2:])), 1)
            mask = torch.cat(
                (mask, mask.new_zeros(b, p)), 1)
        return inputs, mask

    
    @torch.jit.ignore
    def forward(self, inputs, audio, mask, audio_mask,
            audio_lengths:Optional[Tensor]=None,
            prenet_dropout:Optional[float]=None,
            chunk_pad_text:int|None=None,
            chunk_pad_audio:int|None=None,
            temperature:float=1
            ):
        r"""Train Decoder with teacher forcing.
        Args:
            inputs: raw or encoded text.
            audio: audio frames for teacher-forcing.
            mask: text mask for sequence padding.
            audio_mask: audio mask for loss computation.
            prenet_dropout: if not None, override original value
                (to implement e.g. annealing)
            temperature: no effect on training, only on returned output/MSE
        Shapes:
            - inputs: 
                FloatTensor (B, T_text, D_text)
                or LongTensor (B, T_text)
            - audio: (B, T_audio, D_audio)
            - mask: (B, T_text)
            - audio_mask: (B, T_audio)
            - stop_target TODO

            - outputs: (B, T_audio, D_audio)
            - alignments: (B, T_audio, T_text)
            - stop_tokens: (B, T_audio)

        """
        if chunk_pad_audio:
            audio, audio_mask = self.chunk_pad(
                audio, audio_mask, chunk_pad_audio)
        
        if audio_lengths is None:
            audio_lengths = audio_mask.sum(-1).cpu()
        ground_truth = audio
        if prenet_dropout is None:
            prenet_dropout = self.prenet_dropout

        # print(f'{audio[...,0].min()=}')
        audio = self.latent_map(audio)
        # print(f'{audio[...,0].min()=}')

        if inputs.dtype==torch.long:
            assert self.text_encoder is not None
            assert inputs.ndim==2
            inputs = self.text_encoder.encode(inputs, mask)
        if chunk_pad_text:
            inputs, mask = self.chunk_pad(inputs, mask, chunk_pad_text)

        (
            memory, context, alignment,
            attention_hidden, attention_cell, 
            decoder_hidden, decoder_cell 
        ) = self.core.init_states(inputs)  

        # concat the initial audio frame with training data
        memories = torch.cat((memory[None], audio.transpose(0, 1)))
        memories = self.core.prenet(memories, prenet_dropout)

        # print(f'{inputs.shape=}, {context.shape=}, {memories.shape=}, {alignment.shape=}, {mask.shape=}')
        hidden, contexts, alignments = self.core.decode_loop(
            inputs, context, memories, alignment,
            attention_hidden, attention_cell,
            mask)

        # compute the additional decoder layers 
        hidden, output_params, decoder_hidden, decoder_cell = self.core.decode_post(
            hidden, contexts, memories[:-1].transpose(0,1),
            decoder_hidden, decoder_cell, 
            audio_lengths)
        
        r, outputs = self.run_likelihood(
            audio, audio_mask, output_params, temperature=temperature)

        stop_loss = None
        # TODO
        # stop_tokens = self.predict_stop(hidden, outputs)
        # stop_loss = compute_stop_loss(stop_target, stop_tokens)

        outputs = self.latent_unmap(outputs)

        r.update({
            'text': inputs,
            # 'stop_loss': stop_loss,
            'predicted': outputs,
            'ground_truth': ground_truth,
            'alignment': alignments,
            'params': output_params,
            # 'stop': stop_tokens,
            'audio_mask': audio_mask,
            'text_mask': mask,
            **self.likelihood.metrics(output_params),
            **self.alignment_metrics(alignments, mask, audio_mask)
        })

        return r
    
    def alignment_metrics(self, alignments, mask, audio_mask, t=2):
        """
        alignments: (B, T_audio, T_text)
        mask: (B, T_text)
        """
        # TODO: could normalize concentration by logT and subtract from 1,
        # so it represents a proportion of the entropy 'unused'
        # then could have a cutoff parameter
        # alignment should hit every token: max. entropy of mean token probs
        # alignment should be sharp: min. mean of token entropy
        concentration = []
        concentration_norm = []
        dispersion = []
        # alternatively:
        # max average length of token prob vectors, and of time-curve vectors
        # if using L2, this enourages token energy to concentrate in few token
        # dimensions per vector, but to spread across multiple time steps 
        # concentration_l2 = []
        # dispersion_l2 = []
        for a,mt,ma in zip(alignments, mask, audio_mask):
            a = a[ma][:,mt]
            mean_probs = a.mean(0)
            ent_mean = (mean_probs * mean_probs.clip(1e-7,1).log()).sum()
            # ent_mean = torch.special.entr(mean_probs).sum()
            concentration.append(ent_mean)
            concentration_norm.append(1 + ent_mean / mt.float().sum().log())
            dispersion.append(-(a*a.clip(1e-7,1).log()).sum(-1).mean())
            # dispersion.append(-torch.special.entr(a).sum(-1).mean())
            # concentration_l2.append(-a.pow(t).mean(0).pow(1/t).mean())
            # dispersion_l2.append(-a.pow(t).mean(1).pow(1/t).mean())
        return {
            'concentration': torch.stack(concentration),#.mean(),
            'concentration_norm': torch.stack(concentration_norm),#.mean(),
            'dispersion': torch.stack(dispersion),#.mean(),
            # 'concentration_l2': torch.stack(concentration_l2).mean(),
            # 'dispersion_l2': torch.stack(dispersion_l2).mean()
        }

    @torch.jit.ignore
    def run_likelihood(self, 
            audio, audio_mask, output_params, temperature:float=1):
        r = {}
        m = audio_mask[...,None]
        audio_m = audio*m
        params_m = output_params*m
        # nll = self.likelihood(audio*m, output_params*m)
        # nll = nll.masked_select(audio_mask).mean()

        # NOTE: could improve training performance here?
        #   use audio_mask before likelihood instead of after
        r['nll'] = (
            self.likelihood(audio_m, params_m)
            .masked_select(audio_mask).mean())

        if isinstance(self.likelihood, DiagonalNormalMixture):
            r['nll_fixed'] = (
                self.likelihood(audio_m, params_m, mode='fixed')
                .masked_select(audio_mask).mean())
            r['nll_posthoc'] = (
                self.likelihood(audio_m, params_m, mode='posthoc')
                .masked_select(audio_mask).mean())
            
        with torch.no_grad():
            # return self.likelihood.sample(output_params, temperature=0)
            outputs = self.likelihood.sample(
                output_params, temperature=temperature) # low memory

        d = audio_m - outputs*m
        r['mse'] = (d*d).mean() * m.numel() / m.float().sum()
        return r, outputs
    
    # @torch.jit.ignore
    @torch.jit.export
    def step(self, 
            alignment:Optional[Tensor]=None,
            audio_frame:Optional[Tensor]=None, 
            temperature:float=1.0
        ):
        r"""
        single step of inference.

        optionally supply `alignment` to force the alignments.
        optionally supply `audio_frame` to set the previous frame.

        Args:
            alignment: B x T_text
            audio_frame: B x D_audio
            temperature: optional sampling temperature
        Returns:
            output: B x D_audio
            alignment: B x T_text
            stop_token: None (not implemented)
        """
        alignment, output_params = self.core.step_pre(alignment, audio_frame)
        decoder_output = self.likelihood.sample(
            output_params, temperature=temperature).squeeze(1)
        
        decoder_output = self.core.step_post(alignment, decoder_output)

        # if debug:
        return dict(
            output=decoder_output,
            alignment=alignment,
            stop_prob=torch.tensor(0.),
            params=output_params
        )
        # else: 
            # return decoder_output, alignment, 0

    # TODO rewrite this or don't script it?
    # @torch.jit.export
    @torch.jit.ignore
    def inference(self, inputs, 
            stop:bool=True, 
            max_steps:Optional[int]=None, 
            temperature:float=1.0,
            alignments:Optional[Tensor]=None,
            audio:Optional[Tensor]=None,
            ):
        r"""Decoder inference
        Can supply forced alignments to text;
        Can also supply audio for teacher forcing, 
        to test consistency with the training code or implement prompting
        Args:
            inputs: Text Encoder outputs.
            stop: use stop gate
            max_steps: stop after this many decoder steps
            temperature: for the sampling distribution
            alignments: forced alignment
            audio: forced last-frame
                (will be offset, you don't need to prepend a start frame)
        Shapes:
            - inputs: (B, T_text, D_text)
            - outputs: (B, T_audio, D_audio)
            - alignments: (B, T_text, T_audio)
            - audio: (B, T_audio, D_audio)
            - stop_tokens: (B, T_audio)
        """
        # max_steps = max_steps or self.max_decoder_steps
        max_steps = max_steps if max_steps is not None else self.max_decoder_steps

        self.reset(inputs)

        outputs = []
        if alignments is None:
            alignments = []
            feed_alignment = False
        else:
            feed_alignment = True

        for i in range(alignments.shape[-1]) if feed_alignment else it.count():
            alignment = alignments[:,:,i] if feed_alignment else None
            audio_frame = (
                audio[:,i-1,:] 
                if i>0 and audio is not None and i-1<audio.shape[1] 
                else None)
            r = self.step(
                temperature=temperature, alignment=alignment, audio_frame=audio_frame)
            outputs.append(r['output'])
            if not feed_alignment:
                alignments.append(r['alignment'])

            if stop and r['stop_prob']>self.stop_threshold:
                break
            if len(outputs) == max_steps:
                if stop:
                    print(f"   > Decoder stopped with {max_steps=}")
                break

        outputs = torch.stack(outputs, 1)
        if not feed_alignment:
            alignments = torch.stack(alignments, 1)

        return outputs, alignments

class TacotronCore(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        frame_channels=None, # RAVE latent dim
        dropout=0.1,
        dropout_type='dropout', #'zoneout'
        prenet_type='original', # disabled
        prenet_dropout=0.2,
        prenet_layers=2,
        prenet_size=256,
        prenet_wn=False,
        hidden_dropout=0,
        separate_stopnet=True, # disabled
        rnn_size=1200,
        rnn_bias=True,
        rnn_layers=1,
        noise_channels=0,
        decoder_type='lstm',
        decoder_layers=1,
        decoder_size=None,
        hidden_to_decoder=True,
        memory_to_decoder=False,
        init_proj=1,
        proj_wn=False,
        attn_wn=False,
        learn_go_frame=False,
        pitch_xform=False,
        block_size=2048,
        max_batch=8,
        max_tokens=1024,
        prior_filter_len=11,
        tokens_per_frame=1.0,
        attention_type='dca',
        length_reparam=False
    ):
        """
        Args:
            in_channels (int): number of input (text feature) channels.
            out_channels (int): number of output (likelihood parameter) channels.
            frame_channels (int): number of audio latent channels.
            dropout (float): dropout rate (except prenet).
            prenet_dropout (float): prenet dropout rate.
            max_decoder_steps (int): Maximum number of steps allowed for the decoder. Defaults to 10000.
        """
        super().__init__()
        assert frame_channels is not None

        self.B = max_batch
        self.T = max_tokens

        self.length_reparam = length_reparam
        self.hidden_to_decoder = hidden_to_decoder
        self.memory_to_decoder = memory_to_decoder

        self.block_size = block_size
        self.frame_channels = frame_channels
        self.pitch_xform = pitch_xform
        # self.r_init = r
        # self.r = r
        self.encoder_embedding_dim = in_channels
        self.separate_stopnet = separate_stopnet

        decoder_size = decoder_size or rnn_size

        # model dimensions
        self.rnn_layers = rnn_layers
        self.decoder_layers = decoder_layers
        self.attention_hidden_dim = rnn_size
        self.decoder_rnn_dim = decoder_size
        self.prenet_dim = prenet_size
        self.noise_channels = noise_channels
        # self.attn_dim = 128
        self.p_attention_dropout = dropout
        self.p_decoder_dropout = dropout
        self.dropout_type = dropout_type

        # memory -> |Prenet| -> processed_memory
        prenet_dim = self.frame_channels
        self.prenet_dropout = prenet_dropout
        self.prenet = Prenet(
            prenet_dim, prenet_type, 
            out_features=[self.prenet_dim]*prenet_layers, 
            bias=False, weight_norm=prenet_wn
        )

        self.hidden_dropout = hidden_dropout

        # self.attention_rnn = DropoutRNN(nn.LSTMCell(
            # self.prenet_dim + in_channels, self.attention_hidden_dim, bias=rnn_bias))
        
        # interlayer dropout is given here, recurrent dropout in forward
        # not for a good reason but it's fine
        self.attention_rnn = ResidualRNN( 
            lambda i,h,**kw: DropoutRNN(nn.LSTMCell(i,h,bias=rnn_bias),**kw),
            self.prenet_dim + in_channels + noise_channels, 
            self.attention_hidden_dim, 
            layers=rnn_layers, dropout=dropout
        )

        if attention_type=='dca':
            prior_alpha = tokens_per_frame / (prior_filter_len-1)
            prior_beta = 1-prior_alpha
            self.attention = MonotonicDynamicConvolutionAttention(
                query_dim=self.attention_hidden_dim,
                attention_dim=128,
                prior_filter_len=prior_filter_len,
                alpha=prior_alpha,
                beta=prior_beta
            )
        elif attention_type=='gauss':
            self.attention = GaussianAttention(
                query_dim=self.attention_hidden_dim,
                prior_tokens_per_frame=tokens_per_frame,
                wn=attn_wn
            )

        # a decoder network further processes the text glimpse and main RNN state,
        # which can increase depth, while using optimized RNN kernels on GPU,
        # which the main RNN can't because it is interleaved with attention.
        # so moving parameters from attention RNN to decoder RNN may make training
        # faster (though not inference)
        # it can also be skipped entirely.
        if decoder_type is None:
            self.decoder_rnn = None
        else:
            if decoder_type=='lstm':
                decoder_type = nn.LSTM
            elif decoder_type=='mlp':
                decoder_type = MLPDecoder
            else:
                raise ValueError(decoder_type)
            decoder_in = in_channels
            if hidden_to_decoder:
                decoder_in += self.attention_hidden_dim 
            if memory_to_decoder: 
                decoder_in += self.prenet_dim + noise_channels
            self.decoder_rnn = decoder_type(
                decoder_in, self.decoder_rnn_dim, 
                num_layers=self.decoder_layers,
                bias=rnn_bias, batch_first=True, 
                dropout=0 if decoder_layers==1 else dropout)

        # this linear projection matches the sizes of the hidden state and
        # the likelihood 
        hidden_size = (
            self.attention_hidden_dim 
            if self.decoder_rnn is None 
            else self.decoder_rnn_dim)
        linear_projection = nn.Linear(
            hidden_size + in_channels, 
            out_channels)
        
        with torch.no_grad():
            linear_projection.weight.mul_(init_proj)

        if proj_wn:
            linear_projection = nn.utils.parametrizations.weight_norm(
                linear_projection)
        self.linear_projection = linear_projection
        

        # the stopnet predicts whether the utterance has ended
        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(
                # self.decoder_rnn_dim + self.frame_channels * self.r_init, 1, 
                hidden_size + self.frame_channels, 1, 
                # bias=True, init_gain="sigmoid"
                ),
        )

        # TODO: need to clone these or no?
        # does register_buffer copy?
        # does Parameter copy?
        t_mem = torch.zeros(1, self.frame_channels)
        t_attn = torch.zeros(self.rnn_layers, 1, self.attention_hidden_dim)
        t_dec = torch.zeros(self.decoder_layers, 1, self.decoder_rnn_dim)
        t_ctx = torch.zeros(1, self.encoder_embedding_dim)

        # initial state parameters
        self.learn_go_frame = learn_go_frame
        # if self.learn_go_frame:
        self.go_frame = nn.Parameter(t_mem.clone())
        self.go_attention_hidden = nn.Parameter(t_attn.clone())
        self.go_attention_cell = nn.Parameter(t_attn.clone())
        self.go_decoder_hidden = nn.Parameter(t_dec.clone())
        self.go_decoder_cell = nn.Parameter(t_dec.clone())
        self.go_context = nn.Parameter(t_ctx.clone())

        # print(id(self.go_attention_hidden.data))
        # print(id(self.go_attention_cell.data))

        # state buffers for inference
        self.register_buffer(
            "memory", t_mem.expand(max_batch, -1).clone()) 
        self.register_buffer(
            "context", t_ctx.expand(max_batch, -1).clone())
        self.register_buffer(
            "attention_hidden", t_attn.expand(-1, max_batch, -1).clone())
        self.register_buffer(
            "attention_cell", t_attn.expand(-1, max_batch, -1).clone())
        self.register_buffer(
            "decoder_hidden", t_dec.expand(-1, max_batch, -1).clone())
        self.register_buffer(
            "decoder_cell", t_dec.expand(-1, max_batch, -1).clone())
        self.register_buffer(
            "alignment", torch.zeros(max_batch, max_tokens))
        self.register_buffer(
            "inputs", torch.zeros(max_batch, max_tokens, in_channels))
      
    def init_states(self, inputs):#, keep_states=False):
        """
        return initial states
        """
        B = inputs.size(0)
        # if not keep_states:
        if self.learn_go_frame:
            attention_hidden = self.go_attention_hidden.expand(B, -1)
            attention_cell = self.go_attention_cell.expand(-1, B, -1)
            decoder_hidden = self.go_decoder_hidden.expand(-1, B, -1)
            decoder_cell = self.go_decoder_cell.expand(-1, B, -1)
            context = self.go_context.expand(B, -1)
            memory = self.go_frame.expand(B,-1).clone()
        else:
            attention_hidden = inputs.new_zeros(
                self.rnn_layers, B, self.attention_hidden_dim)
            attention_cell = torch.zeros_like(attention_hidden)
            decoder_hidden = inputs.new_zeros(
                self.decoder_layers, B, self.decoder_rnn_dim)
            decoder_cell = torch.zeros_like(decoder_hidden)
            context = inputs.new_zeros(B, self.encoder_embedding_dim)
            memory = inputs.new_zeros(B, self.frame_channels)           

        alignment = self.attention.init_states(inputs)

        # for t in (memory, context, alignment,
        #     attention_hidden, attention_cell, 
        #     decoder_hidden, decoder_cell):
        #     print(t.shape)

        return (
            memory, context, alignment,
            attention_hidden, attention_cell, 
            decoder_hidden, decoder_cell)
    
    @torch.jit.export
    def reset(self, inputs):
        r"""
        populates buffers with initial states and text inputs

        call with encoded text, before using `step`

        Args:
            inputs: (B, T_text, D_text)

        """
        assert inputs.ndim==3, str(inputs.shape)#f'{inputs.shape=}'
        B = inputs.shape[0]
        T = inputs.shape[1]
        assert B<=self.inputs.shape[0], 'max batch size exceeded'
        assert T<=self.inputs.shape[1], 'max tokens exceeded'

        (
            self.memory[:B], self.context[:B], self.alignment[:B, :T],
            self.attention_hidden[:,:B], self.attention_cell[:,:B], 
            self.decoder_hidden[:,:B], self.decoder_cell[:,:B] 
        ) = self.init_states(inputs)#, keep_states=False)

        self.inputs[:B, :T] = inputs

        self.B = B
        self.T = T

    # TODO: should there be option to include acoustic memory here?
    @torch.jit.export
    def get_state(self):
        return {
            'rnn_states': (
                self.attention_hidden.clone(),
                self.attention_cell.clone(),
                self.decoder_hidden.clone(),
                self.decoder_cell.clone()
            ),
            'context': self.context.clone(),
            'attention_states': self.alignment.clone(),
        }
    
    @torch.jit.export
    def set_state(self, state:Dict[str,Tensor|Tuple[Tensor, Tensor, Tensor, Tensor]]):
        rnn_states = state['rnn_states']
        if torch.jit.is_scripting():
            # weirdly torschript seems to demand this while Python chokes on 
            # the use of generic types with isintance
            assert isinstance(rnn_states, Tuple[Tensor, Tensor, Tensor, Tensor])
        (
            self.attention_hidden[:],
            self.attention_cell[:],
            self.decoder_hidden[:],
            self.decoder_cell[:]
        ) = rnn_states
        context = state['context']
        alignment = state['attention_states']
        assert isinstance(context, Tensor)
        assert isinstance(alignment, Tensor)
        self.context[:] = context
        self.alignment[:] = alignment

    # @torch.jit.export
    def forward(self, 
            inputs,
            context, memory, alignment,
            attention_hidden, attention_cell,
            set_alignment:bool=False,
            mask:Optional[Tensor]=None):
        """run step of attention loop
        Args:
            inputs: [B, T_text, D_text] encoded text
            context: [B, D_text] combined text encoding from previous alignment
            memory: [B, D_audio] acoustic memory of last output
            alignment: [B, T_text]
            attention_hidden: [B, attention_hidden_dim]
            attention_cell: [B, attention_hidden_dim]
            set_alignment: bool
                if True, `alignment` is the next alignment to text
                if False, `alignment` is the previous alignment, 
                    and the attention module computes next
        Returns:
            context: as above
            alignment: as above
            attention_hidden: as above
            attention_cell: as above
        """
        # feed the latest text and audio encodings into the RNN
        query_input = [memory, context]
        if self.noise_channels:
            query_input.append(torch.randn(
                context.shape[0], self.noise_channels, 
                device=context.device, dtype=context.dtype))
        query_input = torch.cat(query_input, -1)
        attention_hidden, attention_cell = self.attention_rnn(
            query_input, (attention_hidden, attention_cell),
            training=self.training, 
            dropout_p=self.p_attention_dropout, 
            dropout_type=self.dropout_type)

        if not set_alignment:
            # compute next alignment from the RNN state
            alignment = self.attention.align(
                attention_hidden[-1], alignment, inputs, mask)

        # combine text encodings according to the new alignment
        context = self.attention.apply(alignment, inputs, mask)

        return (
            context,
            alignment,
            attention_hidden, attention_cell
        )

    @torch.jit.export
    def decode_post(self, 
            hidden, context, memory,
            decoder_hidden, decoder_cell,
            lengths:Optional[Tensor]=None
        ):
        """run post-decoder (step or full time dimension)
        
        Args:
            hidden: B x T_audio x channel (hidden state after attention net)
            context: B x T_audio x D_text (audio-aligned text features)
            lengths: if not None, pack the inputs
        Returns:
            hidden: B x T_audio x channel (hidden state after decoder net)
            output_params: B x T_audio x channel (likelihood parameters)
        """
        if self.decoder_rnn is not None:
            decoder_rnn_input = []
            if self.hidden_to_decoder:
                decoder_rnn_input.append(hidden)
            if self.memory_to_decoder:
                decoder_rnn_input.append(memory)
            decoder_rnn_input.append(context)
            decoder_rnn_input = torch.cat(decoder_rnn_input, -1)
            if lengths is not None:
                decoder_rnn_input_packed = nn.utils.rnn.pack_padded_sequence(
                    decoder_rnn_input, lengths, 
                    batch_first=True, enforce_sorted=False)

                # self.decoder_hidden and self.decoder_cell: B x D_decoder_rnn
                hidden_packed, (decoder_hidden, decoder_cell) = self.decoder_rnn( 
                    decoder_rnn_input_packed, (decoder_hidden, decoder_cell))
            
                hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    hidden_packed, batch_first=True)
                
                # in case pad_packed messes up extra padding
                hidden = torch.cat((hidden, hidden.new_zeros(
                    hidden.shape[0], 
                    context.shape[1]-hidden.shape[1], 
                    hidden.shape[2])
                    ), 1)
            else:
                # TODO why contiguous needed here when decoder_layers > 1?
                hidden, (decoder_hidden, decoder_cell) = self.decoder_rnn( 
                    decoder_rnn_input, (decoder_hidden.contiguous(), decoder_cell.contiguous()))

        if self.hidden_dropout:
            hidden = F.dropout(hidden, float(self.hidden_dropout), self.training)
        # # B x T x (D_decoder_rnn + D_text)
        decoder_hidden_context = torch.cat((hidden, context), dim=-1)
        # B x T x self.frame_channels
        output_params = self.linear_projection(decoder_hidden_context)
        return hidden, output_params, decoder_hidden, decoder_cell

    @torch.jit.export
    def predict_stop(self, decoder_state, output):
        # B x (D_decoder_rnn + (self.r * self.frame_channels))
        stopnet_input = torch.cat((decoder_state, output), dim=-1)
        if self.separate_stopnet:
            stopnet_input = stopnet_input.detach()
        stop_token = self.stopnet(stopnet_input)
        return stop_token
    
    @torch.jit.export
    def latent_map(self, z):
        if self.pitch_xform:
            z = torch.cat((
                hz_to_z(z[...,:1]),
                z[...,1:]
            ), -1) 

        if self.length_reparam:
            m = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
            z = torch.cat((m, z), -1)
        return z
    
    @torch.jit.export
    def latent_unmap(self, z):
        if self.length_reparam:
            m, z = z.split((1,z.shape[-1]-1), dim=-1)
            z = z * (
                m / (torch.linalg.vector_norm(z, dim=-1, keepdim=True)+1e-7))

        if self.pitch_xform:
            z = torch.cat((
                z_to_hz(z[...,:1]),
                z[...,1:]
            ), -1) 
        return z
    
    def decode_loop(self, 
            inputs, context, memories, alignment, att_hidden, att_cell, mask):
        """loop over training data frames, align to text"""
        hidden, contexts, alignments = [], [], []
        for memory in memories[:-1]:
            context, alignment, att_hidden, att_cell = self(
                inputs, context, memory, alignment,
                att_hidden, att_cell, set_alignment=False,
                mask=mask)
            hidden.append(att_hidden[-1])
            contexts.append(context)
            alignments.append(alignment)
        hidden = torch.stack(hidden, 1)
        contexts = torch.stack(contexts, 1)
        alignments = torch.stack(alignments, 1)
        return hidden, contexts, alignments
    
        # @torch.jit.ignore
    @torch.jit.export
    def step_pre(self, 
            alignment:Optional[Tensor]=None,
            audio_frame:Optional[Tensor]=None
        ):
        B, T = self.B, self.T
        if alignment is None:
            alignment = self.alignment[:B, :T]
            set_alignment = False
        else:
            assert alignment.ndim==2, str(alignment.shape)
            set_alignment = True

        if audio_frame is not None:
            self.memory[:B] = self.latent_map(audio_frame)

        # print(self.memory[:B])

        memory = self.prenet(
            self.memory[:B], dropout=float(self.prenet_dropout))
        # if alignment is not None: print(f'DEBUG: {alignment.shape=}, {memory.shape=}')
        (
            self.context[:B], 
            alignment,
            self.attention_hidden[:,:B], self.attention_cell[:,:B]
        ) = self(
            self.inputs[:B, :T], 
            self.context[:B], memory, alignment,
            self.attention_hidden[:,:B], self.attention_cell[:,:B], 
            set_alignment=set_alignment,
            mask=None)
        (
            _, output_params, 
            self.decoder_hidden[:,:B], self.decoder_cell[:,:B] 
        ) = self.decode_post(
            self.attention_hidden[-1,:B,None], self.context[:B,None], 
            memory[:B,None],
            self.decoder_hidden[:,:B], self.decoder_cell[:,:B],
            lengths=None)
        
        return alignment, output_params
    
    @torch.jit.export
    def step_post(self, alignment, decoder_output):
        self.memory[:self.B] = decoder_output
        self.alignment[:self.B,:self.T] = alignment

        decoder_output = self.latent_unmap(decoder_output)
        return decoder_output


class Prenet(nn.Module):
    """Tacotron specific Prenet with an optional Batch Normalization.
    Note:
        Prenet with BN improves the model performance significantly especially
    if it is enabled after learning a diagonal attention alignment with the original
    prenet. However, if the target dataset is high quality then it also works from
    the start. It is also suggested to disable dropout if BN is in use.
        prenet_type == "original"
            x -> [linear -> ReLU -> Dropout]xN -> o
        prenet_type == "bn"
            x -> [linear -> BN -> ReLU -> Dropout]xN -> o
    Args:
        in_features (int): number of channels in the input tensor and the inner layers.
        prenet_type (str, optional): prenet type "original" or "bn". Defaults to "original".
        prenet_dropout (bool, optional): dropout rate. Defaults to True.
        dropout_at_inference (bool, optional): use dropout at inference. It leads to a better quality for some models.
        out_features (list, optional): List of output channels for each prenet block.
            It also defines number of the prenet blocks based on the length of argument list.
            Defaults to [256, 256].
        bias (bool, optional): enable/disable bias in prenet linear layers. Defaults to True.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        in_features,
        prenet_type="original",
        dropout_at_inference=False,
        out_features=[256, 256],
        bias=True,
        weight_norm=False
    ):
        super().__init__()
        self.prenet_type = prenet_type
        self.dropout_at_inference = dropout_at_inference
        in_features = [in_features] + out_features[:-1]
        # if prenet_type == "bn":
        #     self.linear_layers = nn.ModuleList(
        #         [LinearBN(in_size, out_size, bias=bias) for (in_size, out_size) in zip(in_features, out_features)]
        #     )
        # elif prenet_type == "original":
        norm = (
            nn.utils.parametrizations.weight_norm 
            if weight_norm else lambda x:x)
        self.linear_layers = nn.ModuleList([
            norm(nn.Linear(in_size, out_size, bias=bias))
            for (in_size, out_size) in zip(in_features, out_features)
        ])

    def forward(self, x, dropout:float=0.5):
        for linear in self.linear_layers:
            if dropout:
                x = F.dropout(
                    F.relu(linear(x)), p=dropout, 
                    training=self.training or self.dropout_at_inference)
            else:
                x = F.relu(linear(x))
        return x
    

class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, in_size, size, num_layers=4, 
            bias=True, batch_first=True, 
            dropout=0.1):
        super().__init__()
        assert batch_first

        norm = nn.utils.parametrizations.weight_norm 
        net = [norm(nn.Linear(in_size, size, bias))]
        for _ in range(num_layers):
            block = []
            if dropout:
                block.append(nn.Dropout(dropout))
            block.append(nn.LeakyReLU(0.2))
            block.append(norm(nn.Linear(size, size, bias)))
            net.append(Residual(*block))

        net.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*net)
        # self.net = torch.jit.script(nn.Sequential(*net))

    def forward(self, x, states):
        """pseudo-RNN forward"""
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            h = x.data
        else:
            h = x
        h = self.net(h)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.PackedSequence(h, *x[1:])
        else:
            x = h
        return x, states