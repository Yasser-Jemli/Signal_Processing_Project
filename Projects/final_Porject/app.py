import tkinter as tk
import sounddevice as sd
import scipy.signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import speech_recognition as sr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np 
# pip install pocketsphinx


def start_recording():
    global audio, fs
    fs = 44100
    duration = 5  # 5 seconds recording

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    record_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

def stop_recording():
    sd.wait()  # Wait until recording is finished
    y = audio[:, 0]  # If recording is stereo, take only one channel
    y = scipy.signal.lfilter([1, -0.97], [1], y)

    mfccs = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC Representation')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Convert audio data to the required format
    audio_bytes = y.astype(np.int16).tobytes()
    audio_data = sr.AudioData(audio_bytes, fs, 2) 
    
    recognizer = sr.Recognizer()
    audio_text = recognizer.recognize_sphinx(audio_data, language='en-US')
    
    text.delete(1.0, tk.END)
    text.insert(tk.END, audio_text)

    # Specify the path to save the figure
    save_path = "mfcc_representation.png"
    plt.savefig(save_path)

root = tk.Tk()
root.title("Speech-to-Text App")

record_button = tk.Button(root, text="Record", command=start_recording)
record_button.pack()

stop_button = tk.Button(root, text="Stop", command=stop_recording, state=tk.DISABLED)
stop_button.pack()

text = tk.Text(root)
text.pack()

root.mainloop()
