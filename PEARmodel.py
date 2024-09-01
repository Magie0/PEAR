# coding=utf-8
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
import random
import math

from einops import einsum, rearrange, repeat
import torch.nn.init as init

class mlpCompressor(torch.nn.Module):

  def __init__(self, vocab_size, vocab_dim,  hidden_dim, n_layers, ffn_dim, n_heads,
               batch_size):
    super(mlpCompressor, self).__init__()

    self._vocab_size = vocab_size
    self._vocab_dim = vocab_dim
    self._hidden_dim = hidden_dim
    self._scale = hidden_dim // vocab_dim
    self.input_map = torch.nn.Embedding(vocab_size, vocab_dim*2)
    self.output_logit_map = torch.nn.Linear(self._hidden_dim*2, vocab_size)
    
    torch.nn.init.normal_(self.input_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.output_logit_map.weight, 0, 0.01)
    torch.nn.init.normal_(self.output_logit_map.bias, 0, 0.01)
    
    self.batch_size = batch_size 
    l = []

    l.append(BLFB(8, 16*4, 64, batch_size, 8, 16*8, [True, True]))
    l.append(BAFB(2*2, 128, 64, batch_size, 2*2, 128*4, [True, True]))

    self.U = nn.Parameter(torch.ones(1, 16, 1), requires_grad=True)
    
    self.layers = torch.nn.ModuleList(l)

  def forward(self, x, last=False):
    emb = torch.sigmoid(self.input_map(x))

    bs, seq_len, channels = emb.size()  # 获取 emb 的形状
    x = emb.reshape(bs, 16, seq_len * channels // 16)
    #x = x * self.U
    
    x = self.layers[0].full_forward(x)
    x = self.layers[1].full_forward(x)

    x = self.output_logit_map(x)

    return x, emb
  
  def full_loss(self,
                inputs,
                with_grad=True,
                nonpad_mask=None,
                return_acc=False):

    logits, emb = self.forward(inputs[:, :-1])
    logits = logits.transpose(1, 2)
    loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], inputs[:, -1], reduction='mean')
  
    if with_grad:
      loss.backward()

    return loss, logits

r = 2
class RWB(nn.Module):
  def __init__(self, N=5, B=4, D1=3, D2=2):
    super(RWB, self).__init__()

    self.N = N
    self.B = B
    self.D1 = D1
    self.D2 = D2
    self.linear = nn.Linear(D1//r, D2, bias=True)
    self.U = nn.Parameter(torch.normal(0, 0.01, (N, D1, D2//r)), requires_grad=True)
    self.bias = nn.Parameter(torch.normal(0, 0.01, (N, B, D2)), requires_grad=True)

  def forward(self, x):
    act = torch.bmm(x, self.U)
    act = self.linear(act)
    act = act + self.bias
    return act

class MlpBlock(nn.Module):
    def __init__(self, mlp_dim, output_dim):
        super(MlpBlock, self).__init__()
        self.linear1 = nn.Linear(output_dim, mlp_dim,bias=True)
        self.linear2 = nn.Linear(mlp_dim, output_dim,bias=True)
  
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        return x
    
class SMLP(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(SMLP, self).__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-05, elementwise_affine=True)
        self.MLP = MlpBlock(channels_mlp_dim, hidden_dim)
        self.conv1d = nn.Conv1d(in_channels=tokens_mlp_dim, out_channels=tokens_mlp_dim, kernel_size = 1)

    def forward(self, x):
        skip = x
        x = self.conv1d(x)
        x = x + skip
        y = self.norm1(x)
        x = y + self.MLP(y)
        return x
      
class BAFB(torch.nn.Module):

  def __init__(self, branch, vocab_dim, ffn_dim, batch_size, tokens_mlp_dim, channels_mlp_dim,  ea=[True, True], trans=False):

    super(BAFB, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.ffn_dim = ffn_dim
    self.batch_size = batch_size
    self.V_map = RWB(batch_size, branch, vocab_dim, vocab_dim)
    self.layernorm1 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[0])
    self.layernorm2 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[0])
    self.U_map = SMLP(branch, vocab_dim, tokens_mlp_dim, channels_mlp_dim)
    self.trans = trans
    self.ln1 = 1

    self.U_map1 = torch.nn.Linear(branch*vocab_dim, 4096*2, bias=True)
    
    self.V_map1 = torch.nn.Linear(4096*2, branch*vocab_dim, bias=True)
    
    self.layernorm11 = torch.nn.LayerNorm(branch*vocab_dim, eps=1e-05, elementwise_affine=True)
    self.layernorm22 = torch.nn.LayerNorm(branch*vocab_dim, eps=1e-05, elementwise_affine=True)
    self.ln1 = 1

  def full_forward(self, x):
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)

    if self.ln1:
      x = self.layernorm1(x)
    skip = x

    x = self.U_map(x)
    x = self.layernorm2(x)
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)
    x = self.V_map(x)

    x = (skip + x)/2
    x = x.reshape(self.batch_size, 1, self.branch*self.vocab_dim)

    x = self.GMLP(x)

    return x
  
  def GMLP(self, x):
    
    if self.ln1:
      x = self.layernorm11(x)
         
    skip = x

    x = self.U_map1(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map1(x)
    
    x = self.layernorm22(x)
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x)/2


    return x
  
class BLFB(torch.nn.Module):

  def __init__(self, branch, vocab_dim, ffn_dim, batch_size, tokens_mlp_dim, channels_mlp_dim,  ea=[True, True], trans=False):

    super(BLFB, self).__init__()
    self.branch = branch
    self.vocab_dim = vocab_dim
    self.ffn_dim = ffn_dim
    self.batch_size = batch_size
    self.layernorm2 = torch.nn.LayerNorm(vocab_dim, eps=1e-05, elementwise_affine=ea[0])
    self.U_map = SMLP(branch, vocab_dim, tokens_mlp_dim, channels_mlp_dim)
    self.trans = trans
    self.ln1 = 1

    self.U_map1 = torch.nn.Linear(branch*vocab_dim, 4096, bias=True)
    
    self.V_map1 = torch.nn.Linear(4096, branch*vocab_dim, bias=True)
    
    self.layernorm11 = torch.nn.LayerNorm(branch*vocab_dim, eps=1e-05, elementwise_affine=True)
    self.layernorm22 = torch.nn.LayerNorm(branch*vocab_dim, eps=1e-05, elementwise_affine=True)
    self.ln1 = 1

  def full_forward(self, x):
    x = x.reshape(self.batch_size, self.branch, self.vocab_dim)

    skip = x
    
    x = self.U_map(x)
    x = self.layernorm2(x)

    x = (skip + x)/2
    x = x.reshape(self.batch_size, 1, self.branch*self.vocab_dim)

    x = self.GMLP(x)

    return x
  
  def GMLP(self, x):
    
    if self.ln1:
      x = self.layernorm11(x)
         
    skip = x

    x = self.U_map1(x)
    x = torch.nn.functional.gelu(x)
    x = self.V_map1(x)
    
    x = self.layernorm22(x)
    x = torch.nn.functional.gelu(x)
    
    x = (skip + x)/2


    return x