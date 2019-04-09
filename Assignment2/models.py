#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt


# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
  """
  A helper function for producing N identical layers (each with their own parameters).

  inputs:
      module: a pytorch nn.module
      N (int): the number of copies of that module to return

  returns:
      a ModuleList with the copies of the module (the ModuleList is itself also a module)
  """
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Problem 1
class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    if torch.cuda.is_available():
        self.hidden = torch.Tensor(self.num_layers, self.batch_size, self.hidden_size).cuda()
    else:
        self.hidden = torch.Tensor(self.num_layers, self.batch_size, self.hidden_size)

    # define the function of self.Embedding() and self.Linear()
    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

    dim_in = self.emb_size + self.hidden_size
    self.infirhidden = nn.Linear(dim_in, self.hidden_size)
    dim_multi = self.hidden_size + self.hidden_size
    self.inmulhidden = nn.Linear(dim_multi, self.hidden_size)
    self.tanh = nn.Tanh()
    self.dropout = nn.Dropout(1 - self.dp_keep_prob, True)
    self.linear = nn.Linear(self.hidden_size, self.vocab_size)
    self.softmax = nn.Softmax(self.vocab_size)

    self.hidden = self.init_hidden()
    self.init_weights()

    # TODO ========================
  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and output biases to 0 (in place). The embeddings should not use a bias vector.
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
    # in the range [-k, k] where k is the square root of 1/hidden_size
    k0 = math.sqrt(torch.tensor(1 / (self.hidden_size + self.emb_size)))
    kn = math.sqrt(torch.tensor(1 / self.hidden_size * 2))
    torch.nn.init.uniform_(self.infirhidden.weight, -k0, k0)
    torch.nn.init.uniform_(self.infirhidden.bias, -k0, k0)
    torch.nn.init.uniform_(self.inmulhidden.weight, -kn, kn)
    torch.nn.init.uniform_(self.inmulhidden.bias, -kn, kn)

    torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
    torch.nn.init.uniform_(self.linear.weight, -0.1, 0.1)
    torch.nn.init.zeros_(self.linear.bias)

  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    self.hidden = torch.nn.init.zeros_(self.hidden.clone())

    return self.hidden  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
    seq_len = inputs.shape[0]
    if torch.cuda.is_available():
      hidden_input = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda()  # a-> [seq_len,hidden_size]
      x = torch.zeros(self.batch_size, self.emb_size).cuda()
      a = torch.zeros(self.batch_size, self.hidden_size).cuda()
      logits = Variable(torch.zeros(self.seq_len, self.batch_size, self.vocab_size)).cuda()
    else:
      hidden_input = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))  # a-> [seq_len,hidden_size]
      x = torch.zeros(self.batch_size, self.emb_size)
      a = torch.zeros(self.batch_size, self.hidden_size)
      logits = Variable(torch.zeros(self.seq_len, self.batch_size, self.vocab_size))

    x_input = self.dropout(self.embedding(inputs))  # inputs size = seq_len*batch_size*emb_size

    for t in range(seq_len):
      x = x_input[t]
      for i in range(self.num_layers):
        # Calculate update and reset gates
        if i == 0:
          tmp_first = torch.cat((x, hidden[i].clone()),1)  # [batch,in_feature]+[batch,hidden_size]-> [batch,hidden_size+in_featue]
          a = self.infirhidden(tmp_first)  # [batch,hidden_size+in_feature]*[hidden_size+in_feature,hidden_size] ->[batch,hidden_size]
          hidden[i] = self.tanh(a)
          hidden_input[i] = self.dropout(hidden[i].clone())
        else:
          tmp_multi = torch.cat((hidden_input[i - 1], hidden[i].clone()), 1)  # [batch,in_feature]+[batch,hidden_size]-> [batch,hidden_size+in_featue]
          a = self.inmulhidden(tmp_multi)  # [batch,hidden_size+in_feature]*[hidden_size+in_feature,hidden_size] ->[batch,hidden_size]
          hidden[i] = self.tanh(a)
          hidden_input[i] = self.dropout(hidden[i].clone())

      logits[t] = self.linear(hidden_input[i].clone())
    return logits, hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO =======================
    if torch.cuda.is_available():
      logits = Variable(torch.zeros(self.batch_size, self.vocab_size)).cuda()
      hidden_input = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda()
      # x = Variable(torch.zeros(self.batch_size, self.emb_size)).cuda()
      # a = Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda()
      samples = Variable(torch.zeros([generated_seq_len, self.batch_size], dtype=torch.long)).cuda()
    else:
      logits = Variable(torch.zeros(self.batch_size, self.vocab_size))
      hidden_input = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
      # x = Variable(torch.zeros(self.batch_size, self.emb_size))
      # a = Variable(torch.zeros(self.batch_size, self.hidden_size))
      samples = Variable(torch.zeros([generated_seq_len, self.batch_size], dtype=torch.long))

    samples[0] = input
    m = nn.Softmax()

    for seq_i in range(generated_seq_len - 1):
      if torch.cuda.is_available():
        x = self.embedding(samples[seq_i]).cuda()
      else:
        x = self.embedding(samples[seq_i])

      for i in range(self.num_layers):
        if i == 0:
          tmp_first = torch.cat((x, self.hidden[i].clone()), 1)
          a = self.infirhidden(tmp_first)
          hidden[i] = self.tanh(a)
          hidden_input[i] = hidden[i].clone()
        else:
          tmp_multi = torch.cat((hidden_input[i - 1], self.hidden[i].clone()), 1)
          a = self.inmulhidden(tmp_multi)
          hidden[i] = self.tanh(a)
          hidden_input[i] = hidden[i].clone()
      logits = m(self.linear(hidden_input[i]))  # batch_size*vocab_size
      samples[seq_i + 1] = torch.multinomial(logits, 1).squeeze()
    return samples

# Problem 2


class GRU(nn.Module):  # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """

  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()

    # Define hyper parameters
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dp_keep_prob = dp_keep_prob

    # Define embedding, linear and dropout layers
    self.embed = nn.Embedding(vocab_size, emb_size)
    self.linear = nn.Linear(hidden_size, vocab_size)
    self.drop = nn.Dropout(p=1 - self.dp_keep_prob)

    # Define hidden state
    self.hidden = torch.Tensor(self.num_layers, self.batch_size, self.hidden_size)

    # Define parameters of the first layer
    self.Wz_0 = torch.nn.Parameter(torch.Tensor(hidden_size, emb_size))
    self.Uz_0 = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
    self.bz_0 = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
    self.Wr_0 = torch.nn.Parameter(torch.Tensor(hidden_size, emb_size))
    self.Ur_0 = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
    self.br_0 = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
    self.Wh_0 = torch.nn.Parameter(torch.Tensor(hidden_size, emb_size))
    self.Uh_0 = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
    self.bh_0 = torch.nn.Parameter(torch.Tensor(hidden_size, 1))

    # Define parameters of other layers
    self.Wz = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, hidden_size))
    self.Uz = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, hidden_size))
    self.bz = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, 1))
    self.Wr = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, hidden_size))
    self.Ur = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, hidden_size))
    self.br = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, 1))
    self.Wh = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, hidden_size))
    self.Uh = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, hidden_size))
    self.bh = torch.nn.Parameter(torch.Tensor(num_layers - 1, hidden_size, 1))

    self.init_weights_uniform()
    self.init_hidden()

  def init_weights_uniform(self):

    # Initialize weights and biases of embedding and linear layers
    nn.init.uniform_(self.embed.weight, -0.1, 0.1)
    nn.init.uniform_(self.linear.weight, -0.1, 0.1)
    nn.init.zeros_(self.linear.bias)

    # Initialize weights and biases of recurrent units
    k = math.sqrt(torch.tensor(1 / self.hidden_size))

    torch.nn.init.uniform_(self.Wz_0, -k, k)  # Weights of the first layers
    torch.nn.init.uniform_(self.Uz_0, -k, k)
    torch.nn.init.uniform_(self.bz_0, -k, k)
    torch.nn.init.uniform_(self.Wr_0, -k, k)
    torch.nn.init.uniform_(self.Ur_0, -k, k)
    torch.nn.init.uniform_(self.br_0, -k, k)
    torch.nn.init.uniform_(self.Wh_0, -k, k)
    torch.nn.init.uniform_(self.Uh_0, -k, k)
    torch.nn.init.uniform_(self.bh_0, -k, k)

    for layer in range(self.num_layers - 1):
      torch.nn.init.uniform_(self.Wz[layer], -k, k)  # Weights of other layers
      torch.nn.init.uniform_(self.Uz[layer], -k, k)
      torch.nn.init.uniform_(self.bz[layer], -k, k)
      torch.nn.init.uniform_(self.Wr[layer], -k, k)
      torch.nn.init.uniform_(self.Ur[layer], -k, k)
      torch.nn.init.uniform_(self.br[layer], -k, k)
      torch.nn.init.uniform_(self.Wh[layer], -k, k)
      torch.nn.init.uniform_(self.Uh[layer], -k, k)
      torch.nn.init.uniform_(self.bh[layer], -k, k)

  def init_hidden(self):
    self.hidden = torch.nn.init.zeros_(self.hidden)

    return self.hidden  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    if torch.cuda.is_available():
      logits = Variable(torch.zeros(self.seq_len, self.batch_size, self.vocab_size)).cuda()  # Define output
      hidden_drop = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda()  # Define hidden state with dropout for upward transfer
    else:
      logits = Variable(torch.zeros(self.seq_len, self.batch_size, self.vocab_size))  # Define output
      hidden_drop = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))  # Define hidden state with dropout for upward transfer

    inputs = self.drop(self.embed(inputs))  # inputs size = seq_len*batch_size*emb_size

    for t in range(self.seq_len):
      if torch.cuda.is_available():
        x = inputs[t].cuda()
      else:
        x = inputs[t]

      x = torch.transpose(x, 0, 1)
      for i in range(self.num_layers):
        if i == 0:
          hidden_t_1 = torch.transpose(hidden[i], 0, 1).clone()  # Previous hidden state
          # Calculate update and reset gates
          z = torch.sigmoid(torch.mm(self.Wz_0, x) + torch.mm(self.Uz_0, hidden_t_1) + self.bz_0)
          r = torch.sigmoid(torch.mm(self.Wr_0, x) + torch.mm(self.Ur_0, hidden_t_1) + self.br_0)
          # Calculate hidden units
          hidden_hat = torch.tanh(torch.mm(self.Wh_0, x) + torch.mm(self.Uh_0, torch.mul(r, hidden_t_1)) + self.bh_0)
          hidden_t = torch.mul(z, hidden_hat) + torch.mul((1 - z), hidden_t_1)  # hidden_size*batch_size
          hidden[i] = torch.transpose(hidden_t.clone(), 0, 1).clone()
          hidden_drop[i] = torch.transpose(self.drop(hidden_t.clone()), 0, 1).clone()
        else:
          input_t = torch.transpose(hidden_drop[i - 1], 0, 1).clone()
          hidden_t_1 = torch.transpose(hidden[i], 0, 1).clone()
          # Calculate update and reset gates
          z = torch.sigmoid(torch.mm(self.Wz[i - 1], input_t) + torch.mm(self.Uz[i - 1], hidden_t_1) + self.bz[i - 1])
          r = torch.sigmoid(torch.mm(self.Wr[i - 1], input_t) + torch.mm(self.Ur[i - 1], hidden_t_1) + self.br[i - 1])
          # Calculate hidden units
          hidden_hat = torch.tanh(torch.mm(self.Wh[i - 1], input_t) + torch.mm(self.Uh[i - 1], torch.mul(r, hidden_t_1)) + self.bh[i - 1])
          hidden_t = torch.mul(z, hidden_hat) + torch.mul((1 - z), hidden_t_1)  # hidden_size*batch_size
          hidden[i] = torch.transpose(hidden_t.clone(), 0, 1).clone()
          hidden_drop[i] = torch.transpose(self.drop(hidden_t.clone()), 0, 1).clone()
      logits[t] = self.linear(hidden_drop[i].clone())  # batch_size*vocab_size
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    if torch.cuda.is_available():
      logits = Variable(torch.zeros(self.batch_size, self.vocab_size)).cuda()  # Define output of softmax
      hidden_drop = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).cuda()  # Define hidden state with dropout for upward transfer
      samples = Variable(torch.zeros([generated_seq_len, self.batch_size], dtype=torch.long)).cuda()  # Define output
    else:
      logits = Variable(torch.zeros(self.batch_size, self.vocab_size))  # Define output of softmax
      hidden_drop = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))  # Define hidden state with dropout for upward transfer
      samples = Variable(torch.zeros([generated_seq_len, self.batch_size], dtype=torch.long))  # Define output

    samples[0] = input
    m = nn.Softmax()

    for seq_i in range(generated_seq_len - 1):
      if torch.cuda.is_available():
        x = self.drop(self.embed(samples[seq_i])).cuda()  # input size = batch_size*emb_size
      else:
        x = self.drop(self.embed(samples[seq_i]))

      x = torch.transpose(x, 0, 1)
      for i in range(self.num_layers):
        if i == 0:
          hidden_t_1 = torch.transpose(hidden[i], 0, 1).clone()  # Previous hidden state
          # Calculate update and reset gates
          z = torch.sigmoid(torch.mm(self.Wz_0, x) + torch.mm(self.Uz_0, hidden_t_1) + self.bz_0)
          r = torch.sigmoid(torch.mm(self.Wr_0, x) + torch.mm(self.Ur_0, hidden_t_1) + self.br_0)
          # Calculate hidden units
          hidden_hat = torch.tanh(torch.mm(self.Wh_0, x) + torch.mm(self.Uh_0, torch.mul(r, hidden_t_1)) + self.bh_0)
          hidden_t = torch.mul(z, hidden_hat) + torch.mul((1 - z), hidden_t_1)  # hidden_size*batch_size
          hidden[i] = torch.transpose(hidden_t.clone(), 0, 1).clone()
          hidden_drop[i] = torch.transpose(self.drop(hidden_t.clone()), 0, 1).clone()
        else:
          input_t = torch.transpose(hidden_drop[i - 1], 0, 1).clone()
          hidden_t_1 = torch.transpose(hidden[i], 0, 1).clone()
          # Calculate update and reset gates
          z = torch.sigmoid(torch.mm(self.Wz[i - 1], input_t) + torch.mm(self.Uz[i - 1], hidden_t_1) + self.bz[i - 1])
          r = torch.sigmoid(torch.mm(self.Wr[i - 1], input_t) + torch.mm(self.Ur[i - 1], hidden_t_1) + self.br[i - 1])
          # Calculate hidden units
          hidden_hat = torch.tanh(
            torch.mm(self.Wh[i - 1], input_t) + torch.mm(self.Uh[i - 1], torch.mul(r, hidden_t_1)) + self.bh[i - 1])
          hidden_t = torch.mul(z, hidden_hat) + torch.mul((1 - z), hidden_t_1)  # hidden_size*batch_size
          hidden[i] = torch.transpose(hidden_t.clone(), 0, 1).clone()
          hidden_drop[i] = torch.transpose(self.drop(hidden_t.clone()), 0, 1).clone()
      logits = m(self.linear(hidden_drop[i].clone()))  # batch_size*vocab_size
      samples[seq_i + 1] = torch.multinomial(logits, 1).squeeze()  # generated_seq_len, batch_size
    return samples

# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################


"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""


# This code has been modified from an open-source project, by David Krueger.
# The original license is included below:
# MIT License
#
# Copyright (c) 2018 Alexander Rush
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class MultiHeadedAttention(nn.Module):
  def __init__(self, n_heads, n_units, dropout=0.1):
    """
    n_heads: the number of attention heads
    n_units: the number of output units
    dropout: probability of DROPPING units
    """
    super(MultiHeadedAttention, self).__init__()

    # This requires the number of n_heads to evenly divide n_units.
    assert n_units % n_heads == 0

    # This sets the size of the keys, values, and queries (self.d_k) to all
    # be equal to the number of output units divided by the number of heads.
    self.n_units = n_units
    self.d_k = n_units // n_heads
    self.h = n_heads

    # TODO: create/initialize any necessary parameters or layers
    # Initialize all weights and biases uniformly in the range [-k, k],
    # where k is the square root of 1/n_units.
    # Note: the only Pytorch modules you are allowed to use are nn.Linear
    # and nn.Dropout
    self.q_linear = nn.Linear(n_units, n_units)
    self.v_linear = nn.Linear(n_units, n_units)
    self.k_linear = nn.Linear(n_units, n_units)
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(n_units, n_units)

    k = math.sqrt(1 / self.n_units)

    nn.init.uniform_(self.q_linear.weight, -k, k)
    nn.init.uniform_(self.q_linear.bias, -k, k)
    nn.init.uniform_(self.v_linear.weight, -k, k)
    nn.init.uniform_(self.v_linear.bias, -k, k)
    nn.init.uniform_(self.k_linear.weight, -k, k)
    nn.init.uniform_(self.k_linear.bias, -k, k)
    nn.init.uniform_(self.out.weight, -k, k)
    nn.init.uniform_(self.out.bias, -k, k)

  def forward(self, query, key, value, mask=None):
    # TODO: implement the masked multi-head attention.
    # query, key, and value all have size: (batch_size, seq_len, self.n_units)
    # mask has size: (batch_size, seq_len, seq_len)
    # As described in the .tex, apply input masking to the softmax
    # generating the "attention values" (i.e. A_i in the .tex)
    # Also apply dropout to the attention values.

    bs = query.size(0)

    # perform linear operation and split into h heads
    q = self.q_linear(query).view(bs, -1, self.h, self.d_k)
    k = self.k_linear(key).view(bs, -1, self.h, self.d_k)
    v = self.v_linear(value).view(bs, -1, self.h, self.d_k)

    # transpose to get dimensions bs * h * n_units
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # get attention using function we will define next
    scores = self.getAttention(q, k, v, self.d_k, mask, self.dropout)

    # concatenate heads and put through final linear layer
    concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.n_units)
    output = self.out(concat)

    return output

  def getAttention(self, q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      mask = mask.unsqueeze(1)
      scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
      scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output
# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence


class WordEmbedding(nn.Module):
  def __init__(self, n_units, vocab):
    super(WordEmbedding, self).__init__()
    self.lut = nn.Embedding(vocab, n_units)
    self.n_units = n_units

  def forward(self, x):
    # print (x)
    return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
  def __init__(self, n_units, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, n_units)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                         -(math.log(10000.0) / n_units))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)],
                     requires_grad=False)
    return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
  def __init__(self, size, self_attn, feed_forward, dropout):
    super(TransformerBlock, self).__init__()
    self.size = size
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

  def forward(self, x, mask):
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # apply the self-attention
    return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
  """
  This will be called on the TransformerBlock (above) to create a stack.
  """

  def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
    super(TransformerStack, self).__init__()
    self.layers = clones(layer, n_blocks)
    self.norm = LayerNorm(layer.size)

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)


class FullTransformer(nn.Module):
  def __init__(self, transformer_stack, embedding, n_units, vocab_size):
    super(FullTransformer, self).__init__()
    self.transformer_stack = transformer_stack
    self.embedding = embedding
    self.output_layer = nn.Linear(n_units, vocab_size)

  def forward(self, input_sequence, mask):
    embeddings = self.embedding(input_sequence)
    return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
  "Helper: Construct a model from hyperparameters."
  c = copy.deepcopy
  attn = MultiHeadedAttention(n_heads, n_units)
  ff = MLP(n_units, dropout)
  position = PositionalEncoding(n_units, dropout)
  model = FullTransformer(
      transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
      embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
      n_units=n_units,
      vocab_size=vocab_size
  )

  # Initialize parameters with Glorot / fan_avg.
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model


# ----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
  """ helper function for creating the masks. """
  attn_shape = (1, size, size)
  subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(subsequent_mask) == 0


class Batch:
  "Object for holding a batch of data with mask during training."

  def __init__(self, x, pad=0):
    self.data = x
    self.mask = self.make_mask(self.data, pad)

  @staticmethod
  def make_mask(data, pad):
    "Create a mask to hide future words."
    mask = (data != pad).unsqueeze(-2)
    mask = mask & Variable(
        subsequent_mask(data.size(-1)).type_as(mask.data))
    return mask


# ----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
  "layer normalization, as in: https://arxiv.org/abs/1607.06450"

  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
  """
  A residual connection followed by a layer norm.
  Note for code simplicity the norm is first as opposed to last.
  """

  def __init__(self, size, dropout):
    super(ResidualSkipConnectionWithLayerNorm, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
  """
  This is just an MLP with 1 hidden layer
  """

  def __init__(self, n_units, dropout=0.1):
    super(MLP, self).__init__()
    self.w_1 = nn.Linear(n_units, 2048)
    self.w_2 = nn.Linear(2048, n_units)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))
