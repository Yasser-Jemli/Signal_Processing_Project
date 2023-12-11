import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.io import wavfile
from scipy import signal
import threading

# Function to record audio from the microphone and save it to a file
def record_audio(duration, fs, channels, filename):
    def wait_for_enter():
        input()  # Wait for user input
        return

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=int(fs * duration))

    print("Recording... Press Enter to stop.")
    frames = []

    threading.Thread(target=wait_for_enter).start()

    try:
        while True:
            data = stream.read(fs)
            frames.append(np.frombuffer(data, dtype=np.int16))
    except KeyboardInterrupt:
        print("Recording stopped.")
    
    # Save recorded audio to a file
    audio_data = np.hstack(frames)
    wavfile.write(filename, fs, audio_data.astype(np.int16))

    return audio_data

# Function to apply a basic noise gate (for noise reduction)
def noise_gate(signal_data, threshold):
    # Create a mask for values above the threshold
    mask = np.abs(signal_data) > threshold
    # Apply the mask to the signal
    return signal_data * mask

# Function to equalize the voice
def equalize_voice(signal_data):
    # Implement equalization or other voice enhancement techniques here
    # For example, applying an equalizer filter
    # This is a basic example using a simple high-pass filter
    b, a = signal.butter(4, 1000, 'high', fs=44100)
    return lfilter(b, a, signal_data)

# Function to display figures for 2 seconds and save them
def display_and_save_figure(data, title):
    plt.figure(figsize=(8, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show(block=False)
    plt.pause(2)  # Display the figure for 2 seconds
    plt.close()

# Function to save the processed audio
def save_audio(data, filename, fs):
    wavfile.write(filename, fs, data.astype(np.int16))

# Main function
def main():
    duration = 3  # Duration of recording in seconds
    fs = 44100  # Sampling frequency
    channels = 1  # Number of audio channels
    filename = 'recorded_audio.wav'  # Set your desired filename here

    # Record audio and save to file
    audio_data = record_audio(duration, fs, channels, filename)

    # Display and save raw audio figure
    display_and_save_figure(audio_data, "Raw Audio")

    # Apply noise gate for noise reduction
    threshold = 500  # Adjust this threshold as needed
    processed_data = noise_gate(audio_data, threshold)

    # Display and save noise-reduced audio figure
    display_and_save_figure(processed_data, "Noise-Reduced Audio")

    # Equalize the voice
    enhanced_data = equalize_voice(processed_data)

    # Display and save voice-enhanced figure
    display_and_save_figure(enhanced_data, "Voice Enhanced Audio")

    # Save processed audio
    save_audio(enhanced_data, 'processed_audio.wav', fs)

if __name__ == "__main__":
    main()
