from music21 import *
from loadmidi import load_midi_files_from
import midistuff
from modelmaker import Brain
import pandas as pd
from gan import GAN

def main():
    dir           = 'Stevie'
    data_file     = './datasaves/' + dir + '_save'
    gen_file      = './generated_files/' + dir + '_gen'
    model_file    = './saved_models/' + dir + '_model'
    show_gen      = True
    training_mode = True

    if training_mode:
        try:
            t_data = midistuff.mus_seqs_load(data_file)
        except:
            t = load_midi_files_from('./' + dir)

            t_seqs = []
            for seq in t:
                t_seqs.append(midistuff.get_sequences(seq))

            t_data = []
            for seq in t_seqs:
                t_data.append(midistuff.mus_seq_to_data(seq))

            midistuff.mus_seqs_save(t_data, data_file)

        rnn = Brain(t_data,
                    gpu=False,
                    opt='rmsprop',
                    temperature=0.25,
                    train_seq_length=50,
                    num_lstm_layers=5,
                    num_dense_layers=2,
                    lstm_nodes=256,
                    dense_nodes=512,
                    dropout_rate=0.5,
                    learning_rate=0.005,
                    epsilon=0.5,
                    gen_mode='midi',
                    loss_func='sparse_categorical_crossentropy')

        rnn.train(num_of_epochs=10)
        rnn.save(model_file)
    else:
        rnn = Brain(gpu=False,
                    opt='rmsprop',
                    temperature=0.25,
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

    rnn.load(model_file)
    generated_score = rnn.generate()

    midistuff.write_to_midi(generated_score, gen_file)
    if show_gen:
        generated_score.show()

    # gan = GAN(t_data, train_seq_length=50, generate_length=100, temperature=0.7)
    # gan.train(discriminator_epochs=10, generator_epochs=20)
    # generated_score = gan.generate()
    # midistuff.write_to_midi(generated_score, gen_file)
    # if show_gen:
    #     generated_score.show()

if __name__ == "__main__": main()
