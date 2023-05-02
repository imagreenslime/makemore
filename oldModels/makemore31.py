import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import numpy as np

words = open('names.txt' , 'r').read().lower().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {s:i for i,s in stoi.items()}
vocab_size = len(itos)

# adding data to dataset
block_size = 3
def build_dataset(words, block_size):
    block_size = block_size
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X,Y

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self,x):
        print(x.shape, self.weight.shape)
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

    def __init__(self,dim,eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum 
        self.training = True 
        # parameters (back propogation)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        # forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch var
        else:
            xmean = self.running_mean
            xvar = self.running_var 
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # nomrlaize to unit variance
        self.out = self.gamma * xhat + self.beta
        # buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
import random
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size)
Xdev, Ydev = build_dataset(words[n1:n2], block_size)
Xte, Yte = build_dataset(words[n2:], block_size)


nuerons = 10 # embedding vectors
hidden = 100 # nuerons in hidden layer

g = torch.Generator().manual_seed(2147483647)

C = torch.randn((vocab_size, nuerons), generator=g)
# without tanh paddings all the linear layers would collapse into 1 equation
layers = [
    Linear(nuerons * block_size, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    Linear(hidden, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    Linear(hidden, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    Linear(hidden, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    Linear(hidden, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    Linear(hidden, vocab_size, bias=False), BatchNorm1d(vocab_size)
]

with torch.no_grad():
    # last layer
    layers[-1].gamma *= 0.1
    # apply gain
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 20000
batchsize = 32
lossi = []
# training
for i in range(max_steps):
    # mini batch; stochastic gradient descent
    ix = torch.randint(0,Xtr.shape[0], (batchsize,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass 
    emb = C[Xb] # one-hot the characters
    x = emb.view(emb.shape[0], -1) # concat the vectors
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None 
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats
    lossi.append(loss.log10().item())
    if i % 1000 == 0:
        print(i, loss.item())
        
# visualize histograms
plt.figure(figsize=(20,4)) # width and height
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out
        # print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')
# plt.show()
# plt.close()    
# dev/training

# put layers into eval mode (needed for batch norm)
for layer in layers:
    layer.training = False
# evaluate loss
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] 
  x = emb.view(emb.shape[0], -1)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')


# sample from model
for _ in range(20):
    
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        logits = layer(x)
        probs = F.softmax(logits, dim=1)

        ix = torch.multinomial(probs, num_samples=1, generator=g).item()

        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
