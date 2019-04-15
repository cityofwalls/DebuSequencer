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

        self.generator = Brain( data,
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
        dataset_len = len(self.X[0])
        print(dataset_len)
        #self.generator.train(num_of_epochs=generator_epochs)

        # Update self.X and self.y to be arrays of real sequences (with target 1)
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                # Changing the input sequences to span from 1-end with the actual output appended to the end
                # The target of each of these training steps is 1
                self.X[i][j] = np.append(self.X[i][j][1:], self.y[i][j])
                self.y[i][j] = 1

        #print(self.X)
        #print(self.y)

        super().train(num_of_epochs=discriminator_epochs)

        # Update self.X and self.y to be arrays of fake sequences with generated output from the generator (with target 0)
        generated_values = []
        for i in range(dataset_len):
            self.generator.generate()
            generated_values.append(self.generator.last_prediction)

        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                
                print(self.X[i][j])


    # def train(self, num_of_epochs=1):
    #     for epoch in range(num_of_epochs):
    #         self.generator.train(num_of_epochs=1)
    #         self.generator.generate()
    #         generated_value = self.generator.current_predicition
    #         print()
    #         print(self.seed)    # Current seed used to predict
    #         print(self.generator.real_next_after_seed)  # Real value next in sequence
    #         print(generated_value)  # Generator's current prediction
    #
    #         # Train self on actual value (as True or 1 in output)
    #         # and generator's prediction (as False or 0 in output)
    #         actual_X, actual_y = np.append(self.seed[0][0][1:], self.generator.real_next_after_seed), np.array([1])
    #         fake_X, fake_y     = np.append(self.seed[0][0][1:], generated_value), np.array([0])
    #
    #         actual_X, actual_y = actual_X.reshape(1, 1, len(actual_X)), actual_y.reshape(1, 1, 1)
    #         fake_X, fake_y = fake_X.reshape(1, 1, len(fake_X)), fake_y.reshape(1, 1, 1)
    #
    #         self.model.fit(actual_X, actual_y, shuffle=False, verbose=1)
    #         self.model.fit(fake_X, fake_y, shuffle=False, verbose=1)
    #
    #         # Reset seed for next epoch
    #         self.generator.set_seed(self.generator.cat_data)
    #         self.seed = self.generator.seed
    #         self.real_next_after_seed = self.generator.real_next_after_seed


# class GAN:
#     def __init__(self,
#                  data,
#                  gan_generate_length=100,
#                  train_seq_length=10,
#                  lstm_nodes=256,
#                  dense_nodes=512,
#                  dropout_rate=0.3,
#                  learning_rate=0.00005,
#                  epsilon=0.5,):
#         self.data = data
#         self.train_seq_length = train_seq_length
#         self.learning_rate = learning_rate
#         self.epsilon = epsilon
#         self.gan_generate_length = gan_generate_length
#         self.set_generator()
#
#         self.discriminator = Sequential()
#
#         self.discriminator.add(LSTM(lstm_nodes,
#                                     input_shape=(self.generator.X.shape[1], self.generator.X.shape[2]),
#                                     return_sequences=True))
#         self.discriminator.add(Dropout(dropout_rate))
#         self.discriminator.add(LSTM(lstm_nodes, return_sequences=True))
#         self.discriminator.add(Dropout(dropout_rate))
#         self.discriminator.add(LSTM(lstm_nodes, return_sequences=True))
#         self.discriminator.add(Dropout(dropout_rate))
#         self.discriminator.add(LSTM(lstm_nodes, return_sequences=True))
#         self.discriminator.add(Dropout(dropout_rate))
#         self.discriminator.add(LSTM(lstm_nodes, return_sequences=True))
#         self.discriminator.add(Dropout(dropout_rate))
#
#         self.discriminator.add(TimeDistributed(Dense(dense_nodes),
#                                                input_shape=(self.generator.X.shape[1], self.generator.X.shape[2])))
#         self.discriminator.add(Dropout(dropout_rate))
#         self.discriminator.add(TimeDistributed(Dense(dense_nodes),
#                                                input_shape=(self.generator.X.shape[1], self.generator.X.shape[2])))
#         self.discriminator.add(Dropout(dropout_rate))
#
#         self.discriminator.add(TimeDistributed(Dense(self.generator.vocab),
#                                                      input_shape=(self.generator.X.shape[1], self.generator.X.shape[2])))
#
#         self.discriminator.add(Activation('softmax'))
#         self.discriminator.compile(loss='sparse_categorical_crossentropy',
#                                    optimizer=RMSprop(lr=learning_rate, epsilon=epsilon),
#                                    metrics=['accuracy'])
#
#     def set_generator(self):
#         self.generator = Brain(self.data,
#                                generate_length=1,
#                                train_seq_length=self.train_seq_length,
#                                learning_rate=self.learning_rate,
#                                epsilon=self.epsilon)
#         self.seed = self.generator.seed
#         self.real_next_after_seed = self.generator.real_next_after_seed
#
#
#     def train(self, epochs=10):
#         for epoch in range(1, epochs+1):
#             print('\nDiscriminator Epoch {}\n'.format(epoch))
#
#             self.generator.train(num_of_epochs=1)
#             self.generator.generate()
#             generated_value = self.seed + self.generator.current_predicition
#             generated_value = generated_value.reshape(1, len(self.seed) + 1, 1)
#             real_value = array(self.seed + self.real_next_after_seed).reshape(1, len(self.seed) + 1, 1)
#
#             print('\nFitting prediction to actual next value in discriminator\n')
#             self.discriminator.fit(generated_value, real_value, epochs=1, shuffle=False, verbose=1)
#             print('Done')
#
#             print('\nGetting new seed\n')
#             self.generator.set_seed(self.generator.cat_data)
#             self.seed = self.generator.seed
#             self.real_next_after_seed = self.generator.real_next_after_seed
