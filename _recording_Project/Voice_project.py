import numpy as np
import matplotlib.pyplot as plt

import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

# ''' recording wave speech signal '''
# fs = 16000  # Sample rate
# seconds = 3  # Duration of recording
# print('start recording')
# x = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
# write('output.wav', fs, x)  # Save as WAV file

# sd.play(20*x, fs)
# status = sd.wait()  # Wait until file is done playing
# print('fin recording')

# (fs,x) = wav.read("/home/yasser_jemli/Signal_Processing_Project/_recording_Project/output.wav")
# sd.play(20*x, fs) 
# sd.wait()

(fs,x) = wav.read("/home/yasser_jemli/Signal_Processing_Project/_recording_Project/SA1.WAV")
sd.play(20*x, fs)

time=np.arange(0,len(x))/fs; # Time vector on x-axis
plt.figure()
plt.subplot(311)
plt.plot(time, x, label="signal wav")
plt.title('signal wav')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()