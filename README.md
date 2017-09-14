# LSTM and Hierarchical Attention Network

# Goal

In this repo we demonstrate how to build and train two different neural network architectures for *classification of text documents*. We show a simple implementation on an [NC series Data Science Virtual Machine](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) with [Tesla K80 GPUs](http://www.nvidia.com/object/tesla-k80.html), that uses the [Keras API](https://keras.io) for deep learning. Keras is a front end to three of the most popular deep learning frameworks, CNTK, Tensorflow and Theano. 

# Data 

We train this network on documents from the [Amazon reviews data set](https://snap.stanford.edu/data/web-Amazon.html) [*McAuley and Leskovec, Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text, RecSys, 2013*]. The text consists of reviews of different products by users with the labels being the ratings given to the products. We use the extract of this data set from [Zhang et al. 2015](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf).
There are 3,000,000 training, 650,000 test samples and 5 target classes.

# Preprocessing & Initialization

To generate the inputs to the network from the text data, a preprocessing tokenization step is required. Here we use the `Tokenizer` class from Keras which we fit on the most frequent words in the training data set and we also replace infrequent words with a single token. Thus each document can be represented as a vector of word indexes. A truncation / padding with zeros is then applied so that all vectors have equal length. Masking out these zeros can be toggled in the first layer of the network, which is an *embedding* layer (except CNTK which does not support masking yet). 

The initialization of the embedding layer of each network can affect the accuracy of the model and the speed of convergence significantly. To compute an initial embedding, we use [word2vec](https://arxiv.org/pdf/1301.3781.pdf) with skip-grams, since it yields better results than the default random initialization. For a more detailed description of word embeddings and word2vec see, for example, this [tutorial](http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/).


# LSTM 

The first layer in this architecture is an *embedding* layer, which maps each (one-hot encoded) word index to a vector by a linear transformation. Thus each document vector is mapped to a sequence of output vectors via an embedding matrix (which is learned during training). The output of the embedding layer is fed into a *bidirectional LSTM* layer with 100 units (in each direction). The 5-dimensional output is then obtained with a fully connected layer. This network is optimized with stochastic gradient descent using the cross entropy loss. We also use *l2* regularization in all layers.

Using a document length of 300 words and an embedding dimensionality equal to 200, we obtain a model architecture with 1,442,005 trainable weights, of which the large majority resides in the embedding layer.

![model](/images/lstm_model.png)


# Hierarchical Attention Network

This is the architecture proposed in 
[Hierarchical Attention Networks for Document Classiﬁcation, Yang et al. 2016](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). One of its main features is the hierarchical structure, which consists of two levels of bidirectional GRU layers, one for the sequence of words in each sentence, the second for the sequence of sentences in each document. Another feature of the architecture is that it uses an *attention* layer at both the sentence and word levels. The attention mechanism is the one proposed in [Bahdanau et al. 2014](https://arxiv.org/pdf/1409.0473.pdf) and allows for weighting words in each sentence (and sentences in each document) with different degrees of importance according to the context. 

We have implemented the Hierarchical Attention Network in Keras and Theano by adapting 
[Richard Liao's implementation](https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py).
We use a sentence length of 100 words and a document length of 30 sentences. We set the embedding, context and GRU dimensionalities according to the Hierarchical Attention Network paper. We also follow other choices from this paper, that is, initialize the embedding with word2vec; optimize with SGD and momentum; and reorder the documents in the training batches by number of sentences. We also opt to use *l2* regularization in all layers. In this way we obtain an architecture with 4,141,705 trainable weights.

![model](/images/hatt_model.png)

The second layer expands to the following model, which is distributed to all the sentences:

![sent_model](/images/hatt_model_sent.png)


# Performance

We have not fine tuned the hyperparameters, but have tried a few values as an indication. With LSTM we obtain a classification accuracy of 54.7% and with the hierarchical attention network we obtain 59%. However, the latter takes about 10 hours per epoch to train, whereas the former takes less than 3 hours per epoch. Prediction on the test data set takes about 30 minutes for the hierarchical network and 1.7 hours for LSTM.  

Since most of the weights reside in the embedding layer, the training time depends strongly on the size of the vocabulary and the output dimensionality of the embedding. Other factors are the framework (using CNTK is about twice as fast as Tensorflow) and masking (handling of the padded zeros for variable length sequences), which slows down the training. We have also observed that initializing the embedding with word2vec speeds up significantly the convergence to a good value of accuracy.   


# Implementation details

We have trained the models on an Azure NC series Data Science Virtual Machine with Tesla K80 GPUs. In the cases of CNTK and Tensorflow, the framework handles the execution on the GPU automatically. Tensorflow may throw a `ResourceExhaustedError`, due to taking up all the GPU memory. If this error occurs the remedy is to decrease the batch size.

The case of Theano is not so straightforward and requires some manual configuration before executing the code. See the [Theano configuration](http://deeplearning.net/software/theano_versions/0.9.X/library/config.html) and [gpuarray docs](https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29) for more details. In brief, the following steps are required:
1. Ensure the right python dependencies (with `conda install pygpu`)

2. Replace $HOME/.theanorc with:
```
    [global]
    floatX = float32
    device = gpu0
    [lib]
    gpuarray.preallocate=1
```

3. Set the environment variable `THEANO_FLAGS` to `floatX=float32,device=gpu0` and add `/usr/local/cuda-8.0/bin` to the `PATH` environment variable.

