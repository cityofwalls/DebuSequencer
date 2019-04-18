from modelmaker import Brain
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
from keras.utils import np_utils
import numpy as np
from numpy import array
import midistuff

class GAN(Brain):
    def __init__(self,
                 data,
                 gpu=False,
                 train_seq_length=100,
                 num_lstm_layers=5,
                 num_dense_layers=2,
                 lstm_nodes=256,
                 dense_nodes=512,
                 dropout_rate=0.3,
                 temperature=0.5,
                 generate_length=100,
                 num_voices=1,
                 act='softmax',
                 loss_func='sparse_categorical_crossentropy',
                 opt='rmsprop',
                 learning_rate=0.005,
                 epsilon=0.5,
                 gen_mode='midi',
                 header=None):
        Brain.__init__(self,
                       data,
                       gpu=gpu,
                       train_seq_length=train_seq_length,
                       num_lstm_layers=num_lstm_layers,
                       num_dense_layers=num_dense_layers,
                       lstm_nodes=lstm_nodes,
                       dense_nodes=dense_nodes,
                       dropout_rate=dropout_rate,
                       temperature=temperature,
                       generate_length=generate_length,
                       num_voices=num_voices,
                       act=act,
                       loss_func=loss_func,
                       opt=opt,
                       learning_rate=learning_rate,
                       epsilon=epsilon,
                       gen_mode=gen_mode,
                       header=header)

        print('\nVanilla Brain:')
        print(self.model.summary())
        self.model.pop()
        self.model.pop()
        self.model.add(TimeDistributed(Dense(2),
                                        input_shape=(self.X.shape[1], self.X.shape[2])))
        self.model.add(Activation(act))
        self.model.compile(loss=loss_func, optimizer=RMSprop(lr=learning_rate,epsilon=epsilon), metrics=['accuracy'])

        print('\nChanged to Binary Discriminator:')
        print(self.model.summary())

        self.generator = Brain(data,
                               gpu=gpu,
                               train_seq_length=train_seq_length,
                               num_lstm_layers=num_lstm_layers,
                               num_dense_layers=num_dense_layers,
                               lstm_nodes=lstm_nodes,
                               dense_nodes=dense_nodes,
                               dropout_rate=dropout_rate,
                               temperature=temperature,
                               generate_length=1,
                               num_voices=num_voices,
                               act=act,
                               loss_func=loss_func,
                               opt=opt,
                               learning_rate=learning_rate,
                               epsilon=epsilon,
                               gen_mode=gen_mode,
                               header=header)

        self.real_next_after_seed = self.generator.real_next_after_seed

    def train(self, discriminator_epochs=1, generator_epochs=1):
        print('\nTraining generator for {} epochs\n'.format(generator_epochs))
        self.generator.train(num_of_epochs=generator_epochs)

        # Update self.X and self.y to be arrays of real sequences (with target 1)
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                # Changing the input sequences to span from 1-end with the actual output appended to the end
                # The target of each of these training steps is 1
                self.X[i][j] = np.append(self.X[i][j][1:], self.y[i][j])
                self.y[i][j] = 1

        print('\nTraining discriminator on actual data for {} epochs\n'.format(discriminator_epochs))
        super().train(num_of_epochs=discriminator_epochs)

        # Update self.X and self.y to be arrays of fake sequences with generated output from the generator (with target 0)
        print('\nAsking generator to predict {} values'.format(len(self.X)))
        generated_values = []
        for i in range(len(self.X)):
            current_gens = []
            for j in range(len(self.X[i])):
                self.generator.generate()
                current_gens.append(self.generator.last_prediction)
            generated_values.append(current_gens)

        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                self.X[i][j][-1] = generated_values[i][j]
                self.y[i][j] = 0

        print('\nTraining discriminator on generated data for {} epochs\n'.format(discriminator_epochs))
        super().train(num_of_epochs=discriminator_epochs)

    def discriminate(self, cur_seq):
        self.generator.generate()
        gen_prediction = self.generator.last_prediction
        input = np.append(cur_seq[0][0][1:], gen_prediction)
        input = input.reshape(1, 1, len(input))
        y_hat = self.model.predict(input, verbose=0)[0][0]
        return self.sample(y_hat)

    def generate(self):
        predicted_sequence = []
        cur_seq = self.seed
        gen_temperature = self.generator.temperature
        for i in range(self.generate_length):
            y_hat = self.discriminate(cur_seq)
            while y_hat < 0.5:
                self.generator.temperature += 0.05
                y_hat = self.discriminate(cur_seq)
            self.generator.temperature = gen_temperature

            predicted_sequence.append(self.generator.last_prediction)
            cur_seq = np.append(cur_seq[0][0][1:], self.generator.last_prediction)
            cur_seq = cur_seq.reshape(1, 1, len(cur_seq))

        if self.gen_mode == 'midi':
            return midistuff.data_to_mus_seq(predicted_sequence, self.factors, self.num_voices)
        elif self.gen_mode == 'wav':
            return data_to_wav(predicted_sequence, self.header)
