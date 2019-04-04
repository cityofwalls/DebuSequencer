import pandas as pd
import numpy as np
from numpy import array
from random import choice
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import np_utils
import midistuff

class Brain:
    """  """
    def __init__(self,
                 data,
                 train_seq_length=100,
                 num_lstm_layers=2,
                 num_dense_layers=1,
                 lstm_nodes=512,
                 dense_nodes=256,
                 dropout_rate=0.3,
                 temperature=1.0,
                 generate_length=100,
                 num_voices=1,
                 act='softmax',
                 loss_func='sparse_categorical_crossentropy',
                 opt='rmsprop',
                 learning_rate=0.001,
                 epsilon=None):
        """  """
        self.train_seq_length = train_seq_length
        self.X, self.y = self.data_to_X_y(data)

        self.temperature = temperature
        self.generate_length = generate_length
        self.num_voices = num_voices

        self.model = Sequential()
        self.model.add(LSTM(
                        lstm_nodes,
                        input_shape=(self.X.shape[1], self.X.shape[2]),
                        return_sequences=True))
        self.model.add(Dropout(dropout_rate))

        for i in range(1, num_lstm_layers):
            self.model.add(LSTM(lstm_nodes, return_sequences=True))
            self.model.add(Dropout(dropout_rate))

        for i in range(num_dense_layers):
            self.model.add(Dense(dense_nodes))
            self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(self.vocab))
        self.model.add(Activation(act))
        if opt == 'rmsprop':
            self.model.compile(loss=loss_func, optimizer=RMSprop(lr=learning_rate,epsilon=epsilon))
        elif opt == 'adam':
            self.model.compile(loss=loss_func, optimizer=Adam(lr=learning_rate, epsilon=epsilon, amsgrad=False))
        else:
            self.model.compile(loss=loss_func, optimizer=opt)

    def data_to_X_y(self, data):
        """ Given a dataset of music21 objects, get input set (X) and label (y)
            Parameters:  data,               a list of lists where each internal list is a sequence
                                             of music21 objects representing a voice from a midi file.

            Returns:     X,                  a list of lists where each internal list is a sequence
                                             from data of length train_seq_length.

                         y,                  a list of labels (the next value for a given sequence). """

        # Flatten data into a single sequence and factorize
        # This unifies the factors (categories) across all sequences
        flat_data = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                flat_data.append(data[i][j])
        _, factors = pd.factorize(flat_data)
        self.factors = factors
        self.vocab = len(factors)

        cat_data = []
        for i in range(len(data)):
            cat_data.append([])
            for j in range(len(data[i])):
                idx = list(factors).index(data[i][j])
                cat_data[i].append(idx)

        seq_in, seq_out = [], []
        for i in range(len(cat_data)):
            current_seq = cat_data[i]
            for j in range(len(current_seq) - self.train_seq_length):
                seq_in.append(current_seq[j:j+self.train_seq_length])
                seq_out.append(current_seq[j+self.train_seq_length])

        X, y = array(seq_in), array(seq_out)
        X = X.reshape(len(X), 1, len(X[0]))
        y = y.reshape(len(y), 1, 1)

        # Get the seed for generate method
        # self.seed = array(self.get_seed(cat_data, self.train_seq_length))
        # self.seed = self.seed.reshape(1, 1, self.train_seq_length)

        self.set_seed(cat_data)

        return X, y

    def set_seed(self, seq):
        """ Given a categorical sequence, find a random seed sequence of length l. """
        # seq_choice = choice(seq)
        # while len(seq_choice) < self.train_seq_length:
        #     seq_choice = choice(seqs)
        #
        # idx = seq_choice.index(choice(seq_choice[:-self.train_seq_length - 1]))
        # seed = seq_choice[idx:idx+self.train_seq_length]
        #
        # seed = array(seed)
        # self.seed = seed.reshape(1, 1, self.train_seq_length)

        seed = []
        for _ in range(self.train_seq_length):
            seed.append(choice(range(self.vocab)))

        seed = array(seed)
        self.seed = seed.reshape(1, 1, self.train_seq_length)

    def train(self, num_of_epochs=2):
        self.model.fit(self.X, self.y, epochs=num_of_epochs, shuffle=False, verbose=1)

    def sample(self, preds):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / self.temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self):
        predicted_sequence = []
        cur_seq = self.seed
        for i in range(self.generate_length * self.num_voices):
            y_hat = self.model.predict(cur_seq, verbose=0)[0][0]
            idx = self.sample(y_hat)
            predicted_sequence.append(idx)
            cur_seq = cur_seq[0][0][1:]
            cur_seq = np.append(cur_seq, idx)
            cur_seq = cur_seq.reshape(1, 1, self.train_seq_length)

        #print(predicted_sequence)

        return midistuff.data_to_mus_seq(predicted_sequence, self.factors, self.num_voices)
