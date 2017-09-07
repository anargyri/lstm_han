# LSTM and Hierarchical Attention Network

# Goal

In this repo we demonstrate how to build and train two different neural network architectures for *classification of text documents*. We show a simple implementation on an [NC series Data Science Virtual Machine](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) with [Tesla K80 GPUs](http://www.nvidia.com/object/tesla-k80.html), that uses the [Keras API](https://keras.io) for deep learning. Keras is a front end to three of the most popular deep learning frameworks, CNTK, Tensorflow and Theano. 

# Data 

We train this network on documents from the [Amazon reviews data set](https://snap.stanford.edu/data/web-Amazon.html) [*McAuley and Leskovec, Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text, RecSys, 2013*]. The text consists of reviews of different products by users and the labels are the ratings given to the products. We use the extract of this data set from [Zhang et al. 2015](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf).
There are 3,000,000 training and 650,000 test samples and 5 target classes.

# Preprocessing & Initialization

To generate the inputs to the network from the text data, a preprocessing tokenization step is required. Here we use the `<Tokenizer>` class from Keras which we fit on the most frequent words in the training data set and we also replace infrequent words with a single token. Thus each document can be represented as a vector of word indexes. A truncation / padding with zeros is then applied so that all vectors have equal length. Masking out these zeros can be specified in the embedding layer of the network, if one wishes so (except in the case of CNTK). 

The way of initializing the embedding layer of each network can affect the accuracy of the model and the speed of convergence significantly. Thus we use [word2vec](https://arxiv.org/pdf/1301.3781.pdf) with skip-grams to obtain the initial embedding, which performs better than the default random initiailization. 


# LSTM 

The first layer in this architecture is an *embedding* layer, which maps each (one-hot encoded) word index to a vector by a linear transformation. Thus each document vector is mapped to a sequence of output vectors via an embedding matrix We (which is learned during training). The output of the embedding layer is fed into a *bidirectional LSTM* layer with 100 dimensional outputs. The eventual 5-dimensional prediction is generated with a fully connected layer. This network is optimized with stochastic gradient descent using the cross entropy loss. We also use l2 regularization in all layers.

There are 1,442,005 trainable weights in the architecture, of which the large majority resides in the embedding layer.


# Hierarchical Attention Network

# Performance


# Implementation details

There are about 2,000,000 trainable weights in the architecture, of which the large majority resides in the embedding layer. Thus the training time depends heavily on the size of the vocabulary and the output dimensionality of the embedding. Other considerations affecting time are the framework (using CNTK is about twice faster than Tensorflow) and masking (handling of the padded zeros with variable length LSTMs), which slows down the training. 

Order docs by number of words

GPU Theano

