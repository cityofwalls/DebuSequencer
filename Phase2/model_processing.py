import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
from music21 import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from random import choice
from fractions import Fraction

SEQ_LENGTH = 100

TEMPOS = {
    (0,40):0,
    (40,60):1,
    (60,80):2,
    (80,120):3,
    (120,170):4,
    (170,400):5
}

def seq_to_data(sequences):
    """ sequences: A sequence of music events (notes, chords, rests)

        Converts music events to a numpy array of data points containing pitch information
        (grouped in a tuple if there is more than a single pitch), velocity, and duration
        about the current note or chord, and other general information about the current meter.

        Returns seq, conversions: the sequence of data points converted into categorical data
        and a list of conversions to translate data back into note information. """
    seq_full = []
    for i in range(len(sequences)):
        current_seq = sequences[i] # [(pitch(es), vel, dur), inst, tempo, key, time_sig]

        # Unpack all additional sequence information
        current_inst = current_seq[1]
        current_tempo = current_seq[2]
        current_key = current_seq[3]
        current_time_sig = current_seq[4]

        # Unpack note/chord data and formulate into data points
        pitch_data = current_seq[0]
        for j in range(len(pitch_data)):
            # Grab information specific to the note or chord we're looking at
            pitches = pitch_data[j][0]
            velocity = pitch_data[j][1]
            duration = pitch_data[j][2]

            # Combine pitches, tempo, key, time sig, velocity, duration, inst into a single data point
            data_point = (
                pitches,            # [0]
                current_tempo,      # [1]
                current_key,        # [2]
                current_time_sig,   # [3]
                velocity,           # [4]
                duration,           # [5]
                current_inst        # [6]
            )

            seq_full.append(data_point)

    return seq_full

def __get_seq_seed(seq):
    # TODO: come up with a helper function to create a random note sequence if there is a sequence
    #       in seqs longer that SEQ_LENGTH
    seq_choice = choice(seq)
    while len(seq_choice) < SEQ_LENGTH:
        seq_choice = choice(seqs)

    idx = seq_choice.index(choice(seq_choice[:-SEQ_LENGTH - 1]))
    seed = seq_choice[idx:idx+SEQ_LENGTH]
    return seed

def get_X_y_vocab_seed(data):
    """ data: a list of lists that is complete note sequences of all training pieces
        Returns:
        X:      3D numpy array of sequences of length 1 x [number of sequences as time steps] x
                SEQ_LENGTH to use as input to the model
        y:      3D numpy array of single value sequences representing
                the labels of the input data to the model
        vocab:  the count of unique notes found throughout all training music,
                useful for the output layer of a neural network to know
                the number of output classes
        seed:   A randomly selected sequence from one of the sequences in data """
    # Flatten data into a single sequence and factorize
    # This unifies the factors (categories) across all sequences
    flat_data = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            flat_data.append(data[i][j])
    _, factors = pd.factorize(flat_data)
    vocab = len(factors)

    cat_data = []
    for i in range(len(data)):
        cat_data.append([])
        for j in range(len(data[i])):
            idx = list(factors).index(data[i][j])
            cat_data[i].append(idx)

    seq_in, seq_out = [], []
    for i in range(len(cat_data)):
        current_seq = cat_data[i]
        for j in range(len(current_seq) - SEQ_LENGTH):
            seq_in.append(current_seq[j:j+SEQ_LENGTH])
            seq_out.append(current_seq[j+SEQ_LENGTH])

    # Should we convert the seq_in to a tuple instead of a list?
    X, y = array(seq_in), array(seq_out)
    X = X.reshape(len(X), 1, len(X[0]))
    y = y.reshape(len(y), 1, 1)

    # Get the seed for generate
    seed = array(__get_seq_seed(cat_data))
    seed = seed.reshape(1, 1, SEQ_LENGTH)

    return X, y, vocab, seed, factors

def get_model(X, vocab, lstm_nodes=512, dense_nodes=256, dropout_rate=0.3):
    model = Sequential()
    model.add(LSTM(
        lstm_nodes,
        name='DebuSequencer',
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_nodes, name='Bach_size', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_nodes, name='Nodezart', return_sequences=True))
    model.add(Dense(dense_nodes))
    model.add(Dropout(dropout_rate))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return model

def __sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(model, og_seqs, seed, factors, gen_seq_length=100, num_voices=2):
    debusequence = stream.Stream()
    for i in range(num_voices):
        gen = seed
        voice = stream.Stream()
        overall_offset = 0.0
        for j in range(gen_seq_length):
            y_hat = model.predict(gen, verbose=0)[0][0]
            idx = __sample(y_hat)
            y_hat = factors[idx]

            # y_hat is our prediction, turn it back into a note
            if y_hat[0] == '&&':
                new_note = note.Rest()
            elif isinstance(y_hat[0], tuple):
                pitches = []
                for pitch in y_hat[0]:
                    pitches.append(pitch)
                new_note = chord.Chord(pitches)
            else:
                new_note = note.Note(y_hat[0])

            new_note.volume = volume.Volume(velocity=int(y_hat[1]))
            dur = y_hat[5]
            new_note.duration = duration.Duration(dur)
            new_note.offset = overall_offset
            overall_offset += new_note.duration.quarterLength
            new_note.storedInstrument = y_hat[6]

            # insert the prediction into the current voice
            voice.insert(new_note)

            # Remove the first element of our generator seed and append the predicted note
            gen = list(gen[0][0])[1:]
            gen.append(idx)
            gen = array(gen)
            gen = gen.reshape(1, 1, SEQ_LENGTH)

        debusequence.insert(voice)

    return debusequence
