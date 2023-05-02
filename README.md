# make more
done with Andrej Karpathy's neural networking videos
## description
make more trains a neural network to take in a lots of "things" and return something just like that "thing". make more can be fed a database of names or sentences and generate a name or sentence just like it. in the future, i hope to expand it to take in different types of "things" such as audios and what not. 

## technologies
* PyTorch, however many of the basic functions are recreated
* Neural networks; eg back propagation, forward pass, 
* Various methods to make it faster: eg batch norm, wave net, mini batches

## how to use it
if you don't know what a neural network is. it is a system that tries to immitate a human neural network that takes in information and allows it to infer. the model takes in the list of legal names from "names.txt" and runs it through the bigram network which takes n amount of context then the next letter afterwards to determine what the next letter is going to be when generating a new name. 

the model runs the context through the neurons and hidden layers in a forward pass and lets it guess what the next character is going to be. backpropagation is used to determine what tweaks and twists in the weights and biases of the layers what help it determine the right character more accurately.

the model averages around a log loss of 1.9-2.0 at 20,000 steps throughout the neural network. here are some names that are generated.
```python
tybiany. tiffinice. aracela.
lisabino. dan. ikasha.
tevyale. moni. enniquita.
tyla. juleen. sciylee.
javie. sheyland. charyldn.
```
after around 15 minutes of running on my computer, it got to 450,000 steps in the neural network and averages around a log loss of 1.2-1.5. here are some of its samples
```python
denice. lakiya. danette.
penel. elbert. aldese.
leanna. gina. elisabel.
jemellya. nya. gareen.
shauna. betsi. georgee.
blaze. josephine. rondy.
```
