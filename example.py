import os
from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, Masking, GaussianNoise
from keras.optimizers import Adam
from keras_ctcmodel.CTCModel import CTCModel as CTCModel
import pickle
from keras.preprocessing import sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
import numpy as np
from six.moves.urllib.request import urlretrieve

def create_network(nb_features, nb_labels, padding_value):

    # Define the network architecture
    input_data = Input(name='input', shape=(None, nb_features)) # nb_features = image height

    masking = Masking(mask_value=padding_value)(input_data)
    noise = GaussianNoise(0.01)(masking)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(noise)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

    dense = TimeDistributed(Dense(nb_labels + 1, name="dense"))(blstm)
    outrnn = Activation('softmax', name='softmax')(dense)

    network = CTCModel([input_data], [outrnn])
    network.compile(Adam(lr=0.0001))

    return network



if __name__ == '__main__':
    """ Example of recurrent neural network using CTCModel
    applied on sequences of digits. Digits are images from the MNIST dataset that have been concatenated 
    to get observation sequences and label sequences of different lengths (from 2 to 5)."""

    # Download data
    fpath = './seqDigits.pkl'
    origin = 'https://www.dropbox.com/s/or7s6zo038cc01v/seqDigits.pkl?dl=1'    
    if not os.path.exists(fpath):
        print("Downloading data")
        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        urlretrieve(origin, fpath, dl_progress)

    # load data from a pickle file
    (x_train, y_train), (x_test, y_test) = pickle.load(open(fpath, 'rb'))

    nb_labels = 10 # number of labels (10, this is digits)
    batch_size = 32 # size of the batch that are considered
    padding_value = 255 # value for padding input observations
    nb_epochs = 10 # number of training epochs
    nb_train = len(x_train)
    nb_test = len(x_test)
    nb_features = len(x_train[0][0])


    # create list of input lengths
    x_train_len = np.asarray([len(x_train[i]) for i in range(nb_train)])
    x_test_len = np.asarray([len(x_test[i]) for i in range(nb_test)])
    y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])
    y_test_len = np.asarray([len(y_test[i]) for i in range(nb_test)])

    # pad inputs
    x_train_pad = sequence.pad_sequences(x_train, value=float(padding_value), dtype='float32',
                                         padding="post", truncating='post')
    x_test_pad = sequence.pad_sequences(x_test, value=float(padding_value), dtype='float32',
                                        padding="post", truncating='post')
    y_train_pad = sequence.pad_sequences(y_train, value=float(nb_labels),
                                         dtype='float32', padding="post")
    y_test_pad = sequence.pad_sequences(y_test, value=float(nb_labels),
                                        dtype='float32', padding="post")



    # define a recurrent network using CTCModel
    network = create_network(nb_features, nb_labels, padding_value)


    # CTC training
    network.fit(x=[x_train_pad, y_train_pad, x_train_len, y_train_len], y=np.zeros(nb_train), \
                batch_size=batch_size, epochs=nb_epochs)


    # Evaluation: loss, label error rate and sequence error rate are requested
    eval = network.evaluate(x=[x_test_pad, y_test_pad, x_test_len, y_test_len],\
                            batch_size=batch_size, metrics=['loss', 'ler', 'ser'])


    # predict label sequences
    pred = network.predict([x_test_pad, x_test_len], batch_size=batch_size, max_value=padding_value)
    for i in range(10):  # print the 10 first predictions
        print("Prediction :", [j for j in pred[i] if j!=-1], " -- Label : ", y_test[i]) # 