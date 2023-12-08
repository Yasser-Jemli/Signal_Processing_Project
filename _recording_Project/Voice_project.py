import numpy as np
import matplotlib.pyplot as plt

import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

''' recording wave speech signal '''
fs = 16000  # Sample rate
seconds = 3  # Duration of recording
print('start recording')
x = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, x)  # Save as WAV file

sd.play(20*x, fs)
status = sd.wait()  # Wait until file is done playing
print('fin recording')


