from music21 import *
from fractions import Fraction

INSTRUMENT = instrument.Instrument
METRONOME = tempo.MetronomeMark
KEY = key.Key
TIME_SIGNATURE = meter.TimeSignature

def __extract_rest(element):
    dur = element.duration.quarterLength
    if dur > 4.0:
        dur = 4.0
    return '&&', '0', dur

def __extract_note(element):
    return (str(element.pitch.name) +
            str(element.pitch.octave),
            element.volume.velocity,
            element.duration.quarterLength)

def __extract_chord(element):
    current_chord = []
    for i in range(len(element.pitches)):
        current_chord.append(str(element.pitches[i].name) + str(element.pitches[i].octave))
    current_chord = [tuple(current_chord)]
    current_chord.append(element.volume.velocity),
    current_chord.append(element.duration.quarterLength)#__extract_quarter_length(element.duration.quarterLength))
    return current_chord

def __save_sequence(sequences, current_sequence, current_inst, current_metro, current_key, current_time_sig):
    sequence_to_add = [current_sequence]
    if current_inst:
        sequence_to_add.append(current_inst)
    else:
        sequence_to_add.append('Piano')

    if current_metro:
        sequence_to_add.append(current_metro)
    else:
        sequence_to_add.append(90.0)

    if current_key:
        sequence_to_add.append(current_key)
    else:
        sequence_to_add.append(0)

    if current_time_sig:
        sequence_to_add.append(current_time_sig)
    else:
        sequence_to_add.append('4/4')

    sequences.append(sequence_to_add)
    return sequences

def get_all_sequences(lavender):
    sequences = []
    for i in range(len(lavender.parts)):
        part = lavender.parts[i]

        current_sequence = []
        current_inst, current_metro, current_key, current_time_sig = None, None, None, None
        new_sequence = False
        for element in part.recurse():
            # Check is we have new measure information (instrument, tempo, key, time_sig)
            if (isinstance(element, INSTRUMENT) or
                isinstance(element, METRONOME)  or
                isinstance(element, KEY)        or
                isinstance(element, TIME_SIGNATURE)):
            # If yes, we should try to save the previous set of sequences and their measure information AND reset the current sequence
                if new_sequence:
                    sequences = __save_sequence(sequences, current_sequence, current_inst, current_metro, current_key, current_time_sig)
                    new_sequence = False
                    current_sequence = []
            # Also save this new measure information over the old information
                if isinstance(element, INSTRUMENT):
                    current_inst = element.instrumentName
                if isinstance(element, METRONOME):
                    current_metro = element.number
                if isinstance(element, KEY):
                    current_key = element.sharps
                if isinstance(element, TIME_SIGNATURE):
                    current_time_sig = element.ratioString
            # Otherwise we have note/chord data and that should be added to the current sequence
            if isinstance(element, note.Rest):
                new_sequence = True
                current_sequence.append(__extract_rest(element))
            if isinstance(element, note.Note):
                new_sequence = True
                current_sequence.append(__extract_note(element))
            if isinstance(element, chord.Chord):
                new_sequence = True
                current_sequence.append(__extract_chord(element))

        # Save the last sequence outside of the loop
        sequences = __save_sequence(sequences, current_sequence, current_inst, current_metro, current_key, current_time_sig)
    return sequences

def write_to_disk(lavender):
    lavender.write('midi', fp='peepthis.mid')
    peep = converter.parse('./peepthis.mid')
    print('Written to ./peepthis.mid')
