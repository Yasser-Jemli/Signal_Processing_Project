import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import speech_recognition as sr

frequency_sampling, audio_signal = wavfile.read("/home/celadodc-rswl.com/yasser.jamli/Signal_Processing_Project/recorded_audio.wav")

print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')

# This step involves normalizing the signal 
audio_signal = audio_signal / np.power(2, 15)

# In this step, we are extracting the first 100 values from this signal to visualize
audio_signal = audio_signal[:100]
time_axis = 1000 * np.arange(0, len(audio_signal), 1) / float(frequency_sampling)

plt.plot(time_axis, audio_signal, color='blue')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()

# Characterizing the Audio Signal: Transforming to Frequency Domain
print("From Here we start working on Characterizing the Audio Signal")
print("We go Here ... ==> ")

frequency_sampling, audio_signal = wavfile.read("/home/celadodc-rswl.com/yasser.jamli/Signal_Processing_Project/recorded_audio.wav")

# Displaying the parameters like sampling frequency of the audio signal, data type of signal, and its duration
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')

# Normalizing the signal
audio_signal = audio_signal / np.power(2, 15)

# This step involves extracting the length and half length of the signal
length_signal = len(audio_signal)
half_length = np.ceil((length_signal + 1) / 2.0).astype(int)  # Correction here

# Applying mathematics tools for transforming into frequency domain.
signal_frequency = np.fft.fft(audio_signal)

# Normalization of frequency domain signal and squaring it
signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
signal_frequency **= 2

# Extract the length and half length of the frequency transformed signal
len_fts = len(signal_frequency)

# Adjusting the Fourier transformed signal for even as well as the odd case
if length_signal % 2:
    signal_frequency[1:len_fts] *= 2
else:
    signal_frequency[1:len_fts - 1] *= 2

# Extract the power in decibel (dB)
signal_power = 10 * np.log10(signal_frequency)

# Adjust the frequency in kHz for X-axis
x_axis = np.arange(0, half_length, 1) * (frequency_sampling / length_signal) / 1000.0  # Correction here

# Visualizing the characterization of the signal
plt.figure()
plt.plot(x_axis, signal_power, color='black')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Signal power (dB)')
plt.show()

# Feature Extraction from Speech MFCC
frequency_sampling, audio_signal = wavfile.read("/home/celadodc-rswl.com/yasser.jamli/Signal_Processing_Project/recorded_audio.wav")

# Taking first 15000 samples for analysis.
audio_signal = audio_signal[:15000]

# Use the MFCC Feature
features_mfcc = mfcc(audio_signal, frequency_sampling)

# Print the MFCC parameters
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

# Visualize the MFCC feature
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

# Working with the filter bank features
filterbank_features = logfbank(audio_signal, frequency_sampling)

# Print the filterbank features
print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()

# Recognition of Spoken Words
print ("Simple Reconition of Spoken Words ")
recording = sr.Recognizer()
with sr.Microphone() as source: recording.adjust_for_ambient_noise(source)
print("Please Say something:")
audio = recording.listen(source)

try:
   print("You said: \n" + recording.recognize_google(audio))
except Exception as e:
   print(e)

