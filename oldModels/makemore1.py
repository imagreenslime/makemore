words = open('names.txt' , 'r').read().lower().splitlines()

import torch
N = torch.zeros((27,27), dtype=torch.int32)
# map out bi-grams into a 2d array of letters
chars = sorted(set(''.join(words)))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words[:1000]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # model smoothing of 1
        N[ix1, ix2] += 1


# import matplotlib.pyplot as plt

# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off')

# plt.show()


# create word
P = (N + 1).float()
P /= P.sum(1, keepdim=True)


g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

# calculator log likelihood
# minimize log likelihood to 2<
# lower log likelihood = higher accuracy
log_likelihood = 0.0
n = 0
for w in words[:10]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f'{ch1}{ch2}: {prob: .4f} {logprob: .4f}')

# print(log_likelihood)
nll = -log_likelihood
# print(nll/n)

# Neural Net/Deep Learning: Create training set of bigrams (x,y)
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of exmaples: ', num)

import torch.nn.functional as F

# initialize network: randomly generate weights for 27 neurons
W = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent: getting from top to bottom 
for k in range(1):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # one hot encoding
    logits = xenc @ W # log-counts
    # soft max: turns outputs to probabilities
    counts = logits.exp() # equivivalent N
    probs = counts / counts.sum(1, keepdims=True) # probabilities for next char
    loss = -probs[torch.arange(num), ys].log().mean() + 0.5*(W**2).mean()
    # backward pass
    W.grad = None
    loss.backward()
    
    # update: big steps or little steps
    W.data += -300 * W.grad

    
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float() # one hot encoding
        logits = xenc @ W # log-counts
        print(logits)
        counts = logits.exp() # equivivalent N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next char

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break
    print(''.join(out))