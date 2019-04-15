from music21 import *
from loadmidi import load_midi_files_from
import midistuff
from modelmaker import Brain
import pandas as pd
from gan import GAN
import h5py
from keras.models import save_model
from keras.models import load_model

def main():
    dir         = './Bach_Inventions'
    save_file   = './datasaves/' + dir[2:] + '_save.txt'
    gen_file    = './generated_files/' + dir[2:] + '_gen'
    model_file  = './saved_models/' + dir[2:] + '_model' + '.hdf5'
    show_gen    = True

    # t_data = midistuff.mus_seqs_load(save_file)
    # try:
    #     t_data = midistuff.mus_seqs_load(save_file)
    # except:
    t = load_midi_files_from(dir)

    t_seqs = []
    for seq in t:
        t_seqs.append(midistuff.get_sequences(seq))

    t_data = []
    for seq in t_seqs:
        t_data.append(midistuff.mus_seq_to_data(seq))

    midistuff.mus_seqs_save(t_data, save_file)

    # gan = GAN(t_data)
    # gan.train(discriminator_epochs=1, generator_epochs=1)

    rnn = Brain(t_data,
                gpu=False,
                opt='rmsprop',
                temperature=0.25,    # sampling var; 0.1 no entropy (very discrete), 1.0 max entropy (very random)
                train_seq_length=5,
                num_lstm_layers=5,
                num_dense_layers=2,
                lstm_nodes=256,
                dense_nodes=512,
                dropout_rate=0.5,
                learning_rate=0.005,
                epsilon=0.5,
                gen_mode='midi',
                loss_func='sparse_categorical_crossentropy')

    try:
        rnn.model = load_model(
                        model_file,
                        custom_objects=None,
                        compile=True)
    except:
        rnn.train(num_of_epochs=3)

    #out = h5py.File('./saved_models/{}.hdf5'.format(dir[2:] + '_model'),'r').open()

    generated_score = rnn.generate()
    save_model(
        rnn.model,
        './saved_models/{}.hdf5'.format(dir[2:] + '_model'),
        overwrite=True,
        include_optimizer=True
    )
    midistuff.write_to_midi(generated_score, gen_file)
    if show_gen:
        generated_score.show()

if __name__ == "__main__": main()
