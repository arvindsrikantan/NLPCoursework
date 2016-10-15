import numpy as np
from bigram import perplexity
from nltk.tokenize import *
import time 

target_file = 'wiki.model'
# data I/O
data = open('train.txt', 'r').read() # should be simple plain text file
wordss = list(set(word_tokenize(data)))
words = word_tokenize(data)
# print "train words : ",words
time.sleep(3)
data_size, vocab_size = len(words), 1071
print 'data has %d words, %d unique.' % (data_size, vocab_size)
word_to_index = { word:i for i,word in enumerate(wordss) }
index_to_word = { i:word for i,word in enumerate(wordss) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh1 = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden1
Wxh2 = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden2
Whh_f = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden (forward)
Whh_b = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden (backward)
Wh1y = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
Wh2y = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh_f = np.zeros((hidden_size, 1)) # hidden bias (forward)
bh_b = np.zeros((hidden_size, 1)) # hidden bias (backward)
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev_f, hprev_b):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs_f,hs_b, ys, ps = {}, {}, {}, {}, {}
  hs_f[-1] = np.copy(hprev_f)
  hs_b[len(inputs)] = np.zeros_like(hprev_b)
  loss = 0
  # print "len",len(inputs)
  # forward pass the forward hidden layer
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs_f[t] = np.tanh(np.dot(Wxh1, xs[t]) + np.dot(Whh_f, hs_f[t-1]) + bh_f) # hidden state
    
  for t in reversed(xrange(len(inputs))):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs_b[t] = np.tanh(np.dot(Wxh2, xs[t]) + np.dot(Whh_b, hs_b[t+1]) + bh_b)# t+1 should give error, ideally loop should be from len(inputs)-1 to 0	
    ys[t] = np.dot(Wh1y, hs_f[t]) + np.dot(Wh2y, hs_b[t]) + by
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    loss += -np.log(ps[t][targets[t],0])
	
	
	
  # backward pass: compute gradients going backwards
  dWxh1, dWxh2, dWhh_f, dWhh_b, dWh1y, dWh2y = np.zeros_like(Wxh1), np.zeros_like(Wxh2), np.zeros_like(Whh_f), np.zeros_like(Whh_b), np.zeros_like(Wh1y), np.zeros_like(Wh2y)
  dbh1, dbh2, dby = np.zeros_like(bh_f), np.zeros_like(bh_b), np.zeros_like(by)
  dhnext_f, dhnext_b = np.zeros_like(hs_f[0]), np.zeros_like(hs_b[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWh1y += np.dot(dy, hs_f[t].T)
    dby += dy
    dh_f = np.dot(Wh1y.T, dy) + dhnext_f # backprop into h
    dhraw1 = (1 - hs_f[t] * hs_f[t]) * dh_f # backprop through tanh nonlinearity
    dbh1 += dhraw1
    dWxh1 += np.dot(dhraw1, xs[t].T)
    dWhh_f += np.dot(dhraw1, hs_f[t-1].T)
    dhnext_f = np.dot(Whh_f.T, dhraw1)
	
  # backward pass 2
  for t in xrange(len(inputs)):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWh2y += np.dot(dy, hs_b[t].T)
    dh_b = np.dot(Wh2y.T, dy) + dhnext_b # backprop into h
    dhraw2 = (1 - hs_b[t] * hs_b[t]) * dh_b # backprop through tanh nonlinearity
    dbh2 += dhraw2
    dWxh2 += np.dot(dhraw2, xs[t].T)
    dWhh_b += np.dot(dhraw2, hs_b[t+1].T)# will give an error
    dhnext_b = np.dot(Whh_b.T, dhraw2)
  for dparam in [dWxh1 , dWxh2, dWhh_f, dWhh_b, dWh1y, dWh2y, dbh1, dbh2, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh1 , dWxh2, dWhh_f, dWhh_b, dWh1y, dWh2y, dbh1, dbh2, dby, hs_f[len(inputs)-1], hs_b[len(inputs)-1]

n, p = 0, 0
mWxh1, mWxh2, mWhh_f, mWhh_b, mWh1y, mWh2y = np.zeros_like(Wxh1), np.zeros_like(Wxh2), np.zeros_like(Whh_f), np.zeros_like(Whh_b), np.zeros_like(Wh1y), np.zeros_like(Wh2y)
mbh_f, mbh_b, mby = np.zeros_like(bh_f), np.zeros_like(bh_b), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
output = []

"""
Training
"""

while n<=10000:
  seq = 1
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq+1 >= len(data) or n == 0: 
    hprev_f = np.zeros((hidden_size,1)) # reset RNN memory
    hprev_b = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [word_to_index[word] for word in words[p:p+seq_length]]
  targets = [word_to_index[word] for word in words[p+1:p+seq_length+1]]

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh1 , dWxh2, dWhh_f, dWhh_b, dWh1y, dWh2y, dbh1, dbh2, dby, hprev_f, hprev_b = lossFun(inputs, targets, hprev_f, hprev_b)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh1, Wxh2, Whh_f, Whh_b, Wh1y, Wh2y, bh_f, bh_b, by], 
                                [dWxh1, dWxh2, dWhh_f, dWhh_b, dWh1y, dWh2y, dbh1, dbh2, dby], 
                                [mWxh1, mWxh2, mWhh_f, mWhh_b, mWh1y, mWh2y, mbh_f, mbh_b, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq # move data pointer
  n += 1 # iteration counter

open('brnn.model','w').write(str([Wxh1, Wxh2, Whh_f, Whh_b, Wh1y, Wh2y, bh_f, bh_b, by]))

# Wxh1, Wxh2, Whh_f, Whh_b, Wh1y, Wh2y, bh_f, bh_b, by = eval(open('brnn.model').read())
"""
Testing
"""
data = open('test.txt', 'r').read() # should be simple plain text file
words = word_tokenize(data)
observed_words = []
# print "test words : ",words
p=0
time.sleep(4)
n=0
count=0
xs, hs_f,hs_b, ys, ps = {}, {}, {}, {}, {}
while n<=1000:
  seq = 1
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq+1 >= len(words) or n == 0: 
    hprev_f = np.zeros((hidden_size,1)) # reset RNN memory
    hprev_b = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [word_to_index[word] for word in words[p:p+seq_length]]
  targets = [word_to_index[word] for word in words[p+1:p+seq_length+1]]
  # print "inp : ",inputs
  # print "tar : ",[index_to_word[t] for t in targets],"p: ",p
  hs_f[-1] = np.copy(hprev_f)
  hs_b[len(inputs)] = np.zeros_like(hprev_b)
  # forward pass the forward hidden layer
  for t in xrange(len(inputs)):
  	xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
  	xs[t][inputs[t]] = 1
  	hs_f[t] = np.tanh(np.dot(Wxh1, xs[t]) + np.dot(Whh_f, hs_f[t-1]) + bh_f) # hidden state
  hprev_f = hs_f[len(inputs)-1]
  for t in reversed(xrange(len(inputs))):
  	xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
  	xs[t][inputs[t]] = 1
  	hs_b[t] = np.tanh(np.dot(Wxh2, xs[t]) + np.dot(Whh_b, hs_b[t+1]) + bh_b)# t+1 should give error, ideally loop should be from len(inputs)-1 to 0
  	ys[t] = np.dot(Wh1y, hs_f[t]) + np.dot(Wh2y, hs_b[t]) + by
  	ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
  	loss += -np.log(ps[t][targets[t],0])
  # print index_to_word,"ps : ",ps
  print "index : ",ps[len(inputs)-1].flatten().tolist().index(max(ps[len(inputs)-1].flatten())),
  observed = index_to_word[ps[len(inputs)-1].flatten().tolist().index(max(ps[len(inputs)-1].flatten()))]
  observed_words.append(observed)
  print "observed : ",observed, " target : ",index_to_word[targets[-1]]
  if observed == targets[-1]:
    count += 1
  print count
  p += seq # move data pointer
  n += 1 # iter count
print(perplexity(" ".join(observed_words),'wiki.model'))
open("output_sent.txt","w").write(str(observed_words))