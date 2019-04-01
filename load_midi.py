from music21 import *
import glob

def load_lavender():
    lavender_score = converter.parse(
        './Test_Midi/pokemon_lavender.mid')
    return lavender_score

def load_midi_files_from(dir):
    """ dir: folder name with midi files.
        Returns a list of Score objects (music21) from the given directory. """
    lavenders = []
    for file in glob.glob(dir + '/*.mid'):
        lavender_score = converter.parse(file)
        lavenders.append(lavender_score)
    return lavenders
