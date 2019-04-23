import pandas as pd
import numpy as np
from numpy import array
from random import choice
from random import randrange
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import CuDNNGRU
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
import midistuff
from loadwav import data_to_wav
import h5py
from keras.models import save_model
from keras.models import load_model
from fractions import Fraction
import pickle as pkl

class Brain:
    """  """
    def __init__(self,
                 data=None,
                 gpu=False,
                 train_seq_length=100,
                 num_lstm_layers=5,
                 num_dense_layers=2,
                 lstm_nodes=256,
                 dense_nodes=512,
                 dropout_rate=0.3,
                 temperature=0.5,
                 decay=0.7,
                 generate_length=100,
                 num_voices=1,
                 act='softmax',
                 loss_func='sparse_categorical_crossentropy',
                 opt='rmsprop',
                 learning_rate=0.005,
                 epsilon=0.5,
                 gen_mode='midi',
                 header=None):
        """  """
        self.train_seq_length = train_seq_length
        if data:
            self.X, self.y = self.data_to_X_y(data)

        self.temperature = temperature
        self.generate_length = generate_length
        self.last_prediction = None
        self.num_voices = num_voices
        self.gen_mode = gen_mode
        self.header = header

        if data:
            self.model = Sequential()
            if not gpu:
                self.model.add(LSTM(
                                lstm_nodes,
                                input_shape=(self.X.shape[1], self.X.shape[2]),
                                return_sequences=True))
            else:
                self.model.add(CuDNNLSTM(
                                lstm_nodes,
                                input_shape=(self.X.shape[1], self.X.shape[2]),
                                return_sequences=True))
            self.model.add(Dropout(dropout_rate))

            if not gpu:
                for i in range(1, num_lstm_layers):
                    self.model.add(LSTM(lstm_nodes, return_sequences=True))
                    self.model.add(Dropout(dropout_rate))

                for i in range(num_dense_layers):
                    self.model.add(TimeDistributed(Dense(dense_nodes),
                                                         input_shape=(self.X.shape[1], self.X.shape[2])))
                    self.model.add(Dropout(dropout_rate))

                self.model.add(Dense(self.vocab))

            else:
                for i in range(1, num_lstm_layers):
                    self.model.add(CuDNNLSTM(lstm_nodes,
                                             input_shape=(self.X.shape[1], self.X.shape[2]),
                                             return_sequences=True))
                    self.model.add(Dropout(dropout_rate))

                for i in range(num_dense_layers):
                    self.model.add(TimeDistributed(Dense(dense_nodes),
                                                         input_shape=(self.X.shape[1], self.X.shape[2])))
                    self.model.add(Dropout(dropout_rate))

                self.model.add(TimeDistributed(Dense(self.vocab),
                                                     input_shape=(self.X.shape[1], self.X.shape[2])))

            self.model.add(Activation(act))
            if opt == 'rmsprop':
                self.model.compile(loss=loss_func, optimizer=RMSprop(lr=learning_rate, epsilon=epsilon, decay=decay), metrics=['accuracy'])
            elif opt == 'adam':
                self.model.compile(loss=loss_func, optimizer=Adam(lr=learning_rate, epsilon=epsilon, amsgrad=False))
            elif opt == 'sgd':
                self.model.compile(loss=loss_func, optimizer=SGD(lr=0.1, momentum=0.9, decay=0.1, nesterov=False), metrics=['accuracy'])
            else:
                self.model.compile(loss=loss_func, optimizer=opt)
        else:
            self.model = None
            self.cat_data = None
            self.factors = None
            self.vocab = None
            self.seed = array([randrange(100) for i in range(self.train_seq_length)])
            self.seed = self.seed.reshape(1, 1, len(self.seed))

    def data_to_X_y(self, data):
        """ Given a dataset of music21 objects, get input set (X) and label (y)
            Parameters:  data,  a list of lists where each internal list is a sequence
                                of music21 objects representing a voice from a midi file.

            Returns:     X,     a list of lists where each internal list is a sequence
                                from data of length train_seq_length.

                         y,     a list of labels (the next value for a given sequence). """
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

        self.cat_data = cat_data
        self.set_seed(cat_data)

        return X, y

    def set_seed(self, seq):
        """ Given a categorical sequence, find a random seed sequence of length self.train_seq_length. """
        seq_choice = choice(seq)
        while len(seq_choice) < self.train_seq_length:
            seq_choice = choice(seq)

        idx = seq_choice.index(choice(seq_choice[:-self.train_seq_length - 2]))
        seed = seq_choice[idx:idx+self.train_seq_length]

        seed = array(seed)
        self.seed = seed.reshape(1, 1, self.train_seq_length)

        self.real_next_after_seed = seq_choice[idx+self.train_seq_length+1]

        # seed = []
        # for _ in range(self.train_seq_length):
        #     seed.append(choice(range(self.vocab)))
        #
        # seed = array(seed)
        # self.seed = seed.reshape(1, 1, self.train_seq_length)

    def train(self, num_of_epochs=2):
        return self.model.fit(self.X, self.y, epochs=num_of_epochs, shuffle=False, verbose=1)

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
            self.last_prediction = idx
            predicted_sequence.append(idx)
            cur_seq = cur_seq[0][0][1:]
            cur_seq = np.append(cur_seq, idx)
            cur_seq = cur_seq.reshape(1, 1, self.train_seq_length)


        #print(predicted_sequence)
        if self.gen_mode == 'midi':
            return midistuff.data_to_mus_seq(predicted_sequence, self.factors, self.num_voices)
        elif self.gen_mode == 'wav':
            return data_to_wav(predicted_sequence, self.header)

    def save(self, path_and_filename):
        save_model(
            self.model,
            path_and_filename,
            overwrite=True,
            include_optimizer=True
        )

        f = self.factors
        with open(path_and_filename + '.pkl', 'wb') as output:
            pkl.dump(f, output, pkl.HIGHEST_PROTOCOL)

    def load(self, path_and_filename):
        self.model = load_model(path_and_filename,
                               custom_objects=None,
                               compile=True)

        with open(path_and_filename + '.pkl','rb') as input:
            f = pkl.load(input)
        self.factors = f
