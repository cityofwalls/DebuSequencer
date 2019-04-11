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

    rnn = Brain(t_data,
                gpu=True,
                opt='rmsprop',
                temperature=0.4,    # sampling var; 0.1 no entropy (very discrete), 1.0 max entropy (very random)
                train_seq_length=50,
                num_lstm_layers=5,
                num_dense_layers=2,
                lstm_nodes=256,
                dense_nodes=512,
                dropout_rate=0.3,
                learning_rate=0.005,
                epsilon=0.5,
                gen_mode='midi',)
    rnn.train(num_of_epochs=20)

    generated_score = rnn.generate()
    midistuff.write_to_midi(generated_score, 'test')
    generated_score.show()

if __name__ == "__main__": main()
