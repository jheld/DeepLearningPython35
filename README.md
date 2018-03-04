## Overview

### neuralnetworksanddeeplearning.com integrated scripts for Python 3.5.2 and Theano with CUDA support

These scrips are updated ones from the **neuralnetworksanddeeplearning.com** gitHub repository in order to work with Python 3.5.2

The testing file (**test.py**) contains all three networks (network.py, network2.py, network3.py) from the book and it is the starting point to run (i.e. *train and evaluate*) them.

## Just type at shell: **python3.5 test.py**

In test.py there are examples of networks configurations with proper comments. I did that to relate with particular chapters from the book.




## Bubble classification specifics
Please run the following command to setup the samples for the network to be built against:

```python get_circle.py eval_circle.pkl eval_ninety_samples.pkl --random_threshold=0.9 --rand_range 255 256 --dark_cut_off=150 --num_samples=1000 --default_good=1 --default_bad=1 --default_ipsum=1 --default_good_permutations=0 --shuffle_samples=1```

Please run the following command to compile the samples to the correct format:

```python generate_samples.py```


Please run the following command (using any specific hyper-parameters you wish):

```python setup_network.py 40 40 --default_input=1 --monitor_training=1 --eta=0.01 --lmbda=0.0 --binary_classifier=1 --shuffle_input=1 --default_good_permutations=0 --early_stopping_n=5```
