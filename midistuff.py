from music21 import *

INST = instrument.Instrument
MET = tempo.MetronomeMark
KEY = key.Key
TIME_SIG = meter.TimeSignature

def transpose_to_c(note, orig_key):
    if not orig_key:
        return note

    # Traspositions: Each note gets transposed to it's scale degree in C
    # First operate on notes from 'even' keys (number of sharps or flats)
    if orig_key % 2 != 0:
        if orig_key > 0:
            t_pose = (orig_key + 6) % 12
            note.transpose(t_pose)
            return note
        else:
            if orig_key > -6:
                t_pose = orig_key + 6
                note.transpose(-t_pose)
                return note
            else:
                note.transpose(-11)
                return note
    # Otherwise we have an 'odd' number of keys, first look at flats
    elif orig_key < 0:
        if orig_key > -5:
            t_pose = 16 // orig_key
            note.transpose(t_pose)
            return note
        else:
            note.transpose(-6)
            return note
    # We don't change 'even' sharps
    return note

def is_new_measure_info(e):
    return (isinstance(e, INST) or
            isinstance(e, MET)  or
            isinstance(e, KEY)  or
            isinstance(e, TIME_SIG))

def get_element_type(e):
    if isinstance(e, INST):
        return 'inst'
    if isinstance(e, MET):
        return 'met'
    if isinstance(e, KEY):
        return 'key'
    if isinstance(e, TIME_SIG):
        return 'time'
    if isinstance(e, note.Rest):
        return 'rest'
    if isinstance(e, note.Note):
        return 'note'
    if isinstance(e, chord.Chord):
        return 'chord'
    return ''

def get_rest(e):
    dur = e.duration.quarterLength
    if dur > 4.0:
        dur = 4.0
    return '&&', '0', dur

def get_note(e, k):
    c = transpose_to_c(e, k)
    name = str(c.pitch.name) + str(c.pitch.octave)
    vel = c.volume.velocity
    dur = c.duration.quarterLength
    return name, vel, dur

def get_chord(e, k):
    for note in e.pitches:
        note = transpose_to_c(note, k)
    cur_chord = []
    for i in range(len(e.pitches)):
        cur_chord.append(str(e.pitches[i].name) + str(e.pitches[i].octave))
    cur_chord = [tuple(cur_chord)]
    cur_chord.append(e.volume.velocity)
    cur_chord.append(e.duration.quarterLength)
    return cur_chord

def save(seqs, cur_seq, cur_inst, cur_met, cur_key, cur_time):
    sequence_to_add = [cur_seq]
    if cur_inst:
        sequence_to_add.append(cur_inst)
    else:
        sequence_to_add.append('Piano')

    if cur_met:
        sequence_to_add.append(cur_met)
    else:
        sequence_to_add.append(90.0)

    if cur_key:
        sequence_to_add.append(cur_key)
    else:
        sequence_to_add.append(0)

    if cur_time:
        sequence_to_add.append(cur_time)
    else:
        sequence_to_add.append('4/4')

    seqs.append(sequence_to_add)
    return seqs

def get_sequences(lavender):
    sequences = []
    for i in range(len(lavender.parts)):
        part = lavender.parts[i]
        cur_sequence = []
        cur_inst, cur_met, cur_key, cur_time = None, None, None, None
        new_sequence = False
        for element in part.recurse():
            e = get_element_type(element)
            if is_new_measure_info(element):
                if new_sequence:
                    sequences = save(sequences, cur_sequence, cur_inst, cur_met, cur_key, cur_time)
                    new_sequence = False
                    cur_sequence = []

                if e == 'inst':
                    cur_inst = element.instrumentName
                if e == 'met':
                    cur_met = element.number
                if e == 'key':
                    cur_key = element.sharps
                if e == 'time':
                    cur_time = element.ratioString

            if e == 'rest':
                new_sequence = True
                cur_sequence.append(get_rest(element))
            if e == 'note':
                new_sequence = True
                cur_sequence.append(get_note(element, cur_key))
            if e == 'chord':
                new_sequence = True
                cur_sequence.append(get_chord(element, cur_key))

        sequences = save(sequences, cur_sequence, cur_inst, cur_met, cur_key, cur_time)
    return sequences

def mus_seq_to_data(sequences):
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

def data_to_mus_seq(data, factors, num_voices):
    debusequence = stream.Stream()
    for i in range(num_voices):
        voice = stream.Stream()
        offset = 0.0
        for j in range(i*(len(data)//num_voices), (i+1)*(len(data)//num_voices)):
            pred = factors[data[j]]
            if pred[0] == '&&':
                new_note = note.Rest()
            elif isinstance(pred[0], tuple):
                pitches = []
                for pitch in pred[0]:
                    pitches.append(pitch)
                new_note = chord.Chord(pitches)
            else:
                new_note = note.Note(pred[0])

            new_note.volume = volume.Volume(velocity=int(pred[1]))
            new_note.duration = duration.Duration(pred[5])
            new_note.offset = offset
            offset += new_note.duration.quarterLength
            new_note.storedInstrument = pred[6]

            voice.insert(new_note)

        debusequence.insert(voice)

    return debusequence

def write_to_midi(lavender, filename='peepthis'):
    lavender.write('midi', fp='{}.mid'.format(filename))
    peep = converter.parse('./{}.mid'.format(filename))
    print('Written to ./{}.mid'.format(filename))
