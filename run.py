from music21 import *
from load_midi import *
import midi_processing
import model_processing
import midistuff
from modelmaker import Brain

import pandas as pd

def main():
    t = load_midi_files_from('./Bach_Inventions')

    t_seqs = []
    for seq in t:
        t_seqs.append(midistuff.get_sequences(seq))

    t_data = []
    for seq in t_seqs:
        t_data.append(midistuff.mus_seq_to_data(seq))

    rnn = Brain(t_data)
    rnn.train(num_of_epochs=1)

    generated_score = rnn.generate()
    midistuff.write_to_midi(generated_score, 'test')
    generated_score.show()

if __name__ == "__main__": main()
