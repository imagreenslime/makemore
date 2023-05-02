import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import numpy as np
words = open('names.txt' , 'r').read().lower().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {s:i for i,s in stoi.items()}

# adding data to dataset
block_size = 4
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

# training split, dev/validation split, test split
# 80, 10, 10 

# creating data
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1], block_size)
Xdev, Ydev = build_dataset(words[n1:n2], block_size)
Xte, Yte = build_dataset(words[n2:], block_size)

# weights and biases
nuerons = 100 # 50: 2.6, 100: 2.28-2.3, 200: 2.4, 300: 2.8
dimensionEmbeddings = 10 # 5: 2.3, 10: 2.26, 20: 2.24
inputs = dimensionEmbeddings * block_size
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,dimensionEmbeddings), generator=g)
W1 = torch.randn((inputs, nuerons), generator=g)
b1 = torch.randn(nuerons, generator=g)
W2 = torch.randn((nuerons, 27), generator=g)
b2 = torch.randn(27, generator=g)
param = [C,W1, b1, W2, b2]

for p in param:
    p.requires_grad = True

print("num of parameters: ", sum(p.nelement() for p in param))

# training stats
lri = []
lossi = []
stepi = []
max_steps = 30000
batchsize = 32
# training
for i in range(max_steps):
    # mini batch; stochastic gradient descent
    ix = torch.randint(0,Xtr.shape[0], (batchsize,))
    
    # forward pass 
    emb = C[Xtr[ix]] # (x, 3, 2)
    h = torch.tanh(emb.view(-1,inputs) @ W1 + b1) # (x, nuerons)
    logits = h @ W2 + b2 # (x, 27 alphabet)
    counts = logits.exp()
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in param:
        p.grad = None 
    loss.backward()

    # update
    lr = 0.1 if i < 10000 else 0.01
    for p in param:
        p.data += -lr * p.grad

    # stats
    stepi.append(i)
    lossi.append(loss.log10().item())

# dev/training
@torch.no_grad() # decorator disables gradient tracking
def split_loss(split):
    x, y = {
        "train": (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x] # (x, 3, 2)
    h = torch.tanh(emb.view(-1,inputs) @ W1 + b1) # (x, inputs)
    logits = h @ W2 + b2 # (x, 27 alphabet)
    counts = logits.exp()
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss("train")
split_loss("val")

# letter relations
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha='center', va='center', color='white')
    plt.grid('minor')
#plt.show()
#plt.close()    

# plotting step to log loss
plt.plot(stepi, lossi)
#plt.show()
#plt.close()

# sample from model
for _ in range(20):
    
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))
