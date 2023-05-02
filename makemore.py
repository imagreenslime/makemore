import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------
# helper functions for initializing and training the neural network (these functions are basics in torch library, remade for practice)

# initializing the weights and biases
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # initialize the weights and biases of the embedding vectors/nuerons
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self,x):
        # hidden pre layer activation / first forward pass
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

# batch normalization: regularizes the gradient descent and improves training speed 
class BatchNorm1d:
    def __init__(self,dim,eps=1e-5, momentum=0.1):
        self.eps = eps # only for numerical stability
        self.momentum = momentum # exponential moving average
        self.training = True # a non training set indicates the buffers have already been calculated
        # parameters
        self.gamma = torch.ones(dim) # new weights
        self.beta = torch.zeros(dim) # new biases
        # buffers (running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        # forward pass
        if self.training:
            # to calculate an accurate mean and variance of the batch a 2d shape is needed
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
        self.weight = torch.randn((num_embeddings, embedding_dim)) # a lookup table that stores embeddings
    
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

# creates a chain of neural layers in the order it is passed into it thus making the neural network
class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    # calling forward pass
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        # get parameters from all the layers and stretch them into a list
        return [p for layer in self.layers for p in layer.parameters()]

# -------------------------------------------------------------------------------------
# helper functions for debugging and testing

# dev/training
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (trainingX, trainingY),
    'test': (testingX, testingY),
  }[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(f"split: {split}, loss: {loss.item()}")

# sample from model
def modelSample(amount = 3):
    # put layers into eval mode
    for layer in model.layers:
        layer.training = False

    words = []
    for _ in range(amount): 
        out = []
        context = [0] * block_size
        while True:
            # forward pass the neural net
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            # sample from distribution
            ix = torch.multinomial(probs, num_samples=1).item()
            # continue context
            context = context[1:] + [ix]
            out.append(ix)
            # break if we sample "." token
            if ix == 0:
                break
        words.append(''.join(training_set.itos[i] for i in out)) # decode and print word

    # put layers back into training mode
    for layer in model.layers:
        layer.training = True

    return words

# -------------------------------------------------------------------------------------
# creating the databases

class ChararacterDataset:
    def __init__(self, words, chars):
        self.chars = chars
        self.stoi = {s:i+1 for i,s in enumerate(self.chars)}
        self.stoi['.'] = 0
        self.itos = {i:s for s,i in self.stoi.items()}
        self.words = words
        
    # store the context in the first array then the next letter in the second 
    def render_dataset(self):
        X, Y = [], []
        for w in self.words:
            context = [0] * block_size
            for ch in w + ".":
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        X = torch.tensor(X) 
        Y = torch.tensor(Y)
        return(X, Y)

    def __len__(self):
        return len(self.words)

    def getVocabSize(self):
        return len(self.itos)
        
def create_dataset(input_file):
    # read in all the words
    with open(input_file, 'r') as k:
        data = k.read()
    words = data.lower().splitlines()
    random.shuffle(words)
    chars = sorted(list(set(''.join(words))))

    n1 = int(0.8 * len(words)) # 80%
    n2 = int(0.9 * len(words)) # 10%

    training_set = ChararacterDataset(words[:n1], chars)
    testing_set = ChararacterDataset(words[n1:n2], chars)

    return training_set, testing_set

# -------------------------------------------------------------------------------------
# running the nueral network

if __name__ == "__main__":

    block_size = 8 # n context the neural network takes

    training_set, testing_set = create_dataset("names.txt")
    vocab_size = testing_set.getVocabSize()

    trainingX, trainingY = training_set.render_dataset()
    testingX, testingY = testing_set.render_dataset()

    nuerons = 24 # embedding vectors
    hidden = 128 # nuerons in hidden layer

    # g = torch.Generator().manual_seed(2147483647)

    # activate the nueron layers; without tanh paddings all the linear layers would collapse into 1 equation
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
    # activates back propagation
    for p in parameters: 
        p.requires_grad = True

    steps = 0
    batchsize = 32
    lossi = []
    # training
    while True:
        # mini batch; stochastic gradient descent
        ix = torch.randint(0,trainingX.shape[0], (batchsize,))
        Xb, Yb = trainingX[ix], trainingY[ix]
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
        lr = 0.1 if steps < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # stats and samples
        lossi.append(loss.item())
        if steps % 1000 == 0:
            print(f"steps: {steps}, loss: {loss.item()}")

            # sample from model
            models = modelSample(3)
            words = " ".join(models)
            print(f"words generated: {words}")
        steps += 1

# -------------------------------------------------------------------------------------
# debugging and testing

# dev/training
for layer in model.layers:
    layer.training = False

split_loss('train')
split_loss('val')

# visualize steps to loss
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
# plt.show()
# plt.close()

   



