import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

# WaveNet implementation

def create_dataset(input_file):
    # read in all the words
    with open(input_file, 'r') as k:
        data = k.read()
    words = data.lower().splitlines()
    return words

words = create_dataset("names.txt")
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(f"vocab size: {vocab_size}") 

# build dataset
block_size = 8
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X) 
    Y = torch.tensor(Y)
    return(X, Y)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # initialize the weights and biases of the neural network
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self,x):
        # hidden pre layer activation / forward pass
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self,dim,eps=1e-5, momentum=0.1):
        self.eps = eps # numerical stability
        self.momentum = momentum # exponential moving average
        self.training = True # a non training set indicates the buffers have already been calculated
        # parameters (back propogation)
        self.gamma = torch.ones(dim) # weights 2
        self.beta = torch.zeros(dim) # biases 2
        # buffers (running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        # forward pass
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)

            xmean = x.mean(dim, keepdim=True) # mini batch mean 
            xvar = x.var(dim, keepdim=True) # mini batch var
        else:
            xmean = self.running_mean
            xvar = self.running_var 
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize the batch
        self.out = self.gamma * xhat + self.beta

        # update throughout training => each batch
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

# embed and concat characters into vectors
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]
        
class FlattenConsecutive:
    
    def __init__ (self, n):
        self.n = n
        
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n) # split the layer into multiple layers for Wave Net
        if x.shape[1] == 1:
            x = x.squeeze(1) # it removes 1's in the shape
        self.out = x
        return self.out
    
    def parameters(self):
        return []

class Sequential:

    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        # get parameters from all the layers and stretch them into a list
        return [p for layer in self.layers for p in layer.parameters()]

# sample from model
def modelSample(n):
    # put layers into eval mode
    for layer in model.layers:
        layer.training = False

    words = []
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            # forward pass the neural net
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            # sample from distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            # continue context
            context = context[1:] + [ix]
            out.append(ix)
            # break if we sample "." token
            if ix == 0:
                break
        words.append(''.join(itos[i] for i in out)) # decode and print word

    # put layers back into training mode
    for layer in model.layers:
        layer.training = True

    return words

import random
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xte, Yte = build_dataset(words[n1:n2])
Xdev, Ydev = build_dataset(words[n2:])

print(Xtr.shape)
nuerons = 24 # embedding vectors
hidden = 128 # nuerons in hidden layer

g = torch.Generator().manual_seed(2147483647)

# without tanh paddings all the linear layers would collapse into 1 equation
# activate the nueron layers
model = Sequential([
    Embedding(vocab_size, nuerons), 
    FlattenConsecutive(2), Linear(nuerons * 2, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    FlattenConsecutive(2), Linear(hidden * 2, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    FlattenConsecutive(2), Linear(hidden * 2, hidden, bias=False), BatchNorm1d(hidden), Tanh(),
    Linear(hidden, vocab_size),
])

with torch.no_grad():
   model.layers[-1].weight *= 0.1 # last layer less confident

parameters = model.parameters()
print(f"parameter size: {sum(p.nelement() for p in parameters)}")
for p in parameters: 
    p.requires_grad = True

max_steps = 200000
batchsize = 32
lossi = []
# training
for i in range(max_steps):
    # mini batch; stochastic gradient descent
    ix = torch.randint(0,Xtr.shape[0], (batchsize,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    # forward pass 
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for layer in model.layers:
        layer.out.retain_grad() # step learning rate decay
    for p in parameters:
        p.grad = None 
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # stats and samples
    lossi.append(loss.item())
    if i % 1000 == 0:
        print(f"steps: {i}, loss: {loss.item()}")

        # sample from model
        models = modelSample(3)
        words = " ".join(models)
        print(f"words generated: {words}")

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
# plt.show()
# plt.close()

for layer in model.layers:
    layer.training = False

# dev/training
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(f"split: {split}, loss: {loss.item()}")

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
split_loss('train')
split_loss('val')

