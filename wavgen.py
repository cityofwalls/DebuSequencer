from loadwav import load_moro
from loadwav import write_wav
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import *
from random import choice
import pandas as pd

def __sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def wavgen():
    byte_length = 8
    moro, params = load_moro()

    moro = moro.readframes(moro.getnframes())
    header = moro[:44]  # Save the header of the file for later, we don't want to train on this
    data = moro[44:]    # Actual sound information to train on

    data = array(list(data)) / 2.0

    X = []
    y = []
    for i in range(0, len(data) - 800000, byte_length):
        for j in range(i, i + byte_length):
            X.append(data[j])
        y.append(tuple(data[i+byte_length:i+(2*byte_length)]))

    y, uniques = pd.factorize(y)
    num_labels = max(y)
    y = y / num_labels

    X, y = array(X), array(y)
    X = X.reshape(len(X)//byte_length, 1, byte_length)
    y = y.reshape(len(y), 1, 1)

    # Testing model
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(Dense(len(uniques)))
    # model.add(Activation('softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    #
    # model.fit(X, y, epochs=30, shuffle=False, verbose=1)

    # Full model
    model = Sequential()
    model.add(LSTM(
        512,
        name='DebuSequencer',
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, name='Bach_size', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, name='Nodezart', return_sequences=True))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(len(uniques)))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    model.fit(X, y, epochs=3, shuffle=False, verbose=1)

    preds = []
    for i in range(len(X)):
        preds.append(model.predict(X, verbose=0))

    bs = []
    for p in preds[0]:
        current_pred = int(__sample(p))
        for b in uniques[current_pred]:
            bs.append(int(b * 2.0))

    bs = bytes(bs)
    file = header + bs

    print(file)

    write_wav(file, params, './test_wav_write.wav')

if __name__ == "__main__": wavgen()
