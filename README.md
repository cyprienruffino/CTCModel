
# CTCModel : A Connectionnist Temporal Classification implementation for Keras

[![PyPI version](https://badge.fury.io/py/keras-ctcmodel.svg)](https://badge.fury.io/py/keras-ctcmodel)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

CTCModel makes the training of a RNN with the Connectionnist Temporal Classification approach completely transparent.

It directly inherits from the traditionnal Keras Model and uses the TensorFlow implementation of the CTC loss and decoding functions.

## Dependencies
- Keras
- Tensorflow
- six (for the example only)

## Installation
$ pip install keras-ctcmodel

OR

$ git clone https://github.com/cyprienruffino/CTCModel  
$ cd CTCModel
$ python setup.py install --user

## Getting started
Example of a standard recurrent neural network with CTCModel in Keras.

```
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Input
from keras.optimizers import Adam
from numpy import zeros
from keras_ctcmodel.CTCModel import CTCModel as CTCModel

h_features = 10
nb_labels = 10

input_layer = Input((None, h_features))
lstm0 = LSTM(128, return_sequences=True)(input_layer)
lstm1 = LSTM(128, return_sequences=True)(lstm0)
dense = TimeDistributed(Dense(nb_labels))(lstm1)
output_layer = Activation("sigmoid")(dense)

model = CTCModel([input_layer], [output_layer])
model.compile(optimizer=Adam(lr=1e-4))
model.summary()

model.save_model("./")

loaded = CTCModel(None, None)
loaded.load_model("./", optimizer=Adam(lr=1e-4))
loaded.summary()
```


----------


The standard inputs x and y of a Keras Model, where x is the observations and y the labels, are here defined differently. In CTCModel, you must provide as x:

 -  the **input observations**
 -  the **labels**
 -  the **lengths of the input sequences**
 -  the **lengths of the label sequences** 

Here, y is not used in a standard way and must be defined for Keras methods (as the labels or an empty structure of length equal to the length of labels).
Let *x_train*, *y_train*, *x_train_len* and *y_train_len* those terms. Fit, evaluate and predict methods can be used as follow:

```
model.fit(x=[x_train,y_train,x_train_len,y_train_len], y=zeros(nb_train), batch_size=64)
print(model.evaluate(x=[x_test,y_test,x_test_len,y_test_len], batch_size=64))
model.predict([x_test, x_test_len])
```

## Example

The file example.py is an exemple of the use of CTCModel. The dataset is composed of sequence of digits. This is images from the  MNIST datasets [Lecun 98] that have been concatenated to get observation sequences and label sequences.  
The example shows how to use the standard fit, predict and evaluate methods. From the observation and label sequences, we create two list per dataset containing the length of each sequence, one list for the observations and one for the labels. Then data are padded in order to provide inputs of fixed-size to the Keras methods.  

A standard Reccurent Neural Network with bidirectional layers is defined and trained using the *fit* method of CTCModel. Then the *evaluate* method is performed to compute the loss, the label error rate and the sequence error rate on the test set.  The output of the *evaluate* method is thus a list containing the values of each metric. Finally, the *predict* method is applied to get the predictions on the test set. The first predicted sequence are printed in order to compare the predicted labels with the ground truth.  

## Under the hood
CTCModel works by adding three additionnal output layers to a recurrent network for computing the CTC loss, decoding and evaluating using standard metrics for sequence analysis (the sequence error rate and label error rate). Each one can be applied in a blind manner, by the use of standard Keras methods such as *fit*, *predict* and *evaluate*. Note that methods based on generator have been defined and can be used in a standard way, provided that input x and label y that are return by the generator have the specific structure seen above. 

Except the three specific layers, CTCModel works as a standard Keras Model and most of the overriden methods just select the right output layer and call the related Keras Model method. There is also additional methods to save or load model parameters and other ones to get specific computations, e.g. the loss using *get_loss* or the input probabilities using *get_probas* (and the related *on_batch* and *generator* methods). 

## Credits and licence
CTCModel was developped at the LITIS laboratory, Normandie University (http://www.litislab.fr) by Cyprien RUFFINO and Yann SOULLARD, under the supervision of Thierry PAQUET.  

CTCModel is under the terms of the MIT licence.  

CTCModel is hosted on PyPI (https://pypi.org/project/keras-ctcmodel/)

If you use CTCModel for research purposes, please consider adding the following citation to your paper:

<code>
@misc{ctcmodel,
author = {Soullard, Yann and Ruffino, Cyprien and Paquet, Thierry},<br/>
title = {{CTCModel: Connectionist Temporal Classification in Keras}},<br/>
year = {2018},<br/>
ee = {https://arxiv.org/abs/1901.07957},<br/>
archivePrefix = {arXiv}
}
</code>


## References
F. Chollet et al.. Keras: Deep Learning for Python, https://github.com/keras-team/keras, 2015.   
A. Graves, S. Fernández, F. Gomez, J. Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd international conference on Machine learning (pp. 369-376). ACM, June 2006.  
LeCun, Y. (1998). The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/.  
