# Make More
done with Andrej Karpathy's neural networking walk throughs
## description
Make More trains a neural network to take in lots of "things" and return something just like that "thing". Make More can be fed any list of names or sentences generate a name of sentence just like it. 

## technologies
PyTorch however many of the basic functions are recreated
Neural networks; eg: back propagation, forward pass, 
Various methods to make it faster: eg; Batch norm, wave net, mini batches

## how to use it
if you don't know what a neural network is. it is a system that tries to immitate a human neural network that takes in information and allows it to infer. the model takes in the list of legal names from "names.txt" and runs it through the bigram network which takes n amount of context then the next letter afterwards to determine what the next letter is going to be when generating a new name. 

the model runs the context through the neurons and hidden layers in a forward pass and lets it guess what the next character is going to be. backpropagation is used to determine what tweaks and twists in the weights and biases of the layers what help it determine the right character more accurately.

the model averages around a logprob of 1.9-2 at step 20,000. here are some names that are generated.
```python
tybiany. tiffinice. aracela.
lisabino. dan. ikasha.
tevyale. moni. enniquita.
tyla. juleen. sciylee.
javie. sheyland. charyldn.
```
after around 15 minutes of running on my computer, it got to step 450,000 and averages around a logprob 1.2-1.5. here are some of its samples
```python
denice. lakiya. danette.
penel. elbert. aldese.
leanna. gina. elisabel.
jemellya. nya. gareen.
shauna. betsi. georgee.
blaze. josephine. rondy.
```
