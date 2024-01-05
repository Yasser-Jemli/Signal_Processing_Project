import numpy as np
import matplotlib.pyplot as plt

import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

#FFT 
from IPython import get_ipython


# ''' recording wave speech signal '''
# => Code : 
# fs = 16000  # Sample rate
# seconds = 3  # Duration of recording
# print('start recording')
# x = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
# write('output.wav', fs, x)  # Save as WAV file

# play the saved wav 
# sd.play(20*x, fs)
# status = sd.wait()  # Wait until file is done playing
# print('fin recording')

# PLay the saved signal 
# => Code : 
# (fs,x) = wav.read("/home/yasser_jemli/Signal_Processing_Project/_recording_Project/output.wav")
# sd.play(20*x, fs) 
# sd.wait()

# Read the SA1.WAV signal 
(fs,x) = wav.read("/home/yasser_jemli/Signal_Processing_Project/_recording_Project/SA1.WAV")
sd.play(20*x, fs)

# PLot the SA1.WAV signal
time=np.arange(0,len(x))/fs; # Time vector on x-axis
plt.figure()
plt.subplot(311)
plt.plot(time, x, label="signal wav")
plt.title('signal wav')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

#  fft
# ============================================================
get_ipython().magic('reset -sf')

plt.close('all')

# ============================================================
def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n
# ============================================================

# # Calcul de la TFD avec la fft
N  = len(x);
Nf = nextpow2(N)
Fe=fs

X  =np.fft.fft(x,Nf)/N; 
# on effectue un fftshift pour positionner la frequence zero au centre
X  = np.fft.fftshift(X); 
Xa = np.abs(X);

#_ fp=-Fe/2:Fe/Nf:Fe/2-Fe/Nf; ==> # fp=np.arange(-Fe/2,Fe/2,Fe/Nf);
#_ fp=[-Nf/2:Nf/2-1]*Fe/Nf; 
fp=np.arange(-Nf/2,Nf/2)*Fe/Nf;  

# # or
# N = x.size
# Nf = nextpow2(N)
# fp = np.fft.fftshift(np.fft.fftfreq(Nf, d=T0))
 
# -Représentation fréquentiel
plt.figure()
# plt.subplot(312)
plt.stem(fp,Xa)
plt.xlabel('fréquence (Hz)')
plt.ylabel('spectre d''amplitude')