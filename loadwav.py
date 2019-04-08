import wave
import os

def load_moro():
    file = './wav_files_test/Moro_Intro.wav'

    moro = wave.open(file, mode='rb')
    #moro.close()

    return moro, moro.getparams()

def data_to_wav(X, header):
    X__ = bytes(X)

    file = header + X__

    return file

def write_wav(file, params, path):
    output = wave.open(path, mode='wb')
    output.setparams(params)

    #output.writeframesraw(file.readframes(file.getnframes()))
    output.writeframes(file)

    output.close()

    # fd = os.open(path,os.O_RDWR|os.O_CREAT)
    # ret = os.write(fd,file)
    # print("the number of bytes written: ")
    # print(ret)
    # print("written successfully")
    # os.close(fd)
