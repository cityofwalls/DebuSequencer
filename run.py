from music21 import *
from loadmidi import load_midi_files_from
import midistuff
from modelmaker import Brain
import pandas as pd

def main():
    t = load_midi_files_from('./Test_Midi')

    t_seqs = []
    for seq in t:
        t_seqs.append(midistuff.get_sequences(seq))

    t_data = []
    for seq in t_seqs:
        t_data.append(midistuff.mus_seq_to_data(seq))

    rnn = Brain(t_data, opt='rmsprop',
                train_seq_length=10,
                num_lstm_layers=2,
                num_dense_layers=1,
                lstm_nodes=512,
                dense_nodes=256,
                dropout_rate=0.3,
                learning_rate=1e-7,
                epsilon=None,)
    rnn.train(num_of_epochs=50)

    # generated_score = rnn.generate()
    # midistuff.write_to_midi(generated_score, 'test')
    # generated_score.show()

if __name__ == "__main__": main()
