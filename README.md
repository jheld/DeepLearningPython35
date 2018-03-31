## Overview

### neuralnetworksanddeeplearning.com integrated scripts for Python 3.5.2 and Theano with CUDA support

These scrips are updated ones from the **neuralnetworksanddeeplearning.com** gitHub repository in order to work with Python 3.5.2

The testing file (**test.py**) contains all three networks (network.py, network2.py, network3.py) from the book and it is the starting point to run (i.e. *train and evaluate*) them.

## Just type at shell: **python3.5 test.py**

In test.py there are examples of networks configurations with proper comments. I did that to relate with particular chapters from the book.




## Bubble classification specifics
Please run the following command to setup the samples for the network to be built against:

```python generate_samples.py --default_good=1 --default_bad=1 --default_ipsum=1 --default_good_permutations=1 --shuffle_samples=1 --x_size=23 --y_size=23```

Please run the following command to compile the samples to the correct format:

```python format_samples.py --default_good_permutations=1```


Please run the following command (using any specific hyper-parameters you wish):

```python setup_network.py 30 30 --default_input=1 --monitor_training=1 --eta=0.84 --lmbda=0.0 --binary_classifier=1 --shuffle_input=1 --default_good_permutations=1 --early_stopping_n=5 --x_size=23 --y_size=23```

To run the network on a given image:
```python run_evaluation.py network_name image_file_name```