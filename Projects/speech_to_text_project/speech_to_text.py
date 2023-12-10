import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import mfcc, delta
import librosa



audio_file_path = "/home/yasser_jemli/Signal_Processing_Project/Project/SA1.WAV"

def read_audio_file(file_path):
    # Read the audio file
    rate, data = wav.read(file_path)
    return rate, data

def save_audio_file(file_path, rate, data):
    # Save the audio file
    wav.write(file_path, rate, data)

def plot_waveform(signal, title):
    # Plot the waveform
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()

def preemphasis(signal, coeff=0.95):
    # Apply pre-emphasis to the signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, frame_size, frame_stride):
    # Divide the signal into frames
    frame_length, frame_step = frame_size, frame_stride
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    # Pad the signal to ensure that all frames have equal length
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def hamming_window(frame):
    # Apply Hamming window to the frame
    return frame * np.hamming(len(frame))

def compute_mfcc(frame, sample_rate):
    # Compute Mel-frequency cepstral coefficients (MFCCs) for a frame
    n_fft = min(512, len(frame))  # Use the minimum of 512 and the frame length
    hop_length = len(frame) // 2  # Adjust hop_length based on the frame length
    mel_filterbanks = 20
    mfccs = np.mean(librosa.feature.mfcc(y=frame, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=mel_filterbanks).T, axis=0)
    return mfccs

def main(file_path):
    # Read the audio file
    sample_rate, audio_data = read_audio_file(file_path)

    # Apply pre-emphasis
    preemphasized_signal = preemphasis(audio_data)

    # Save pre-emphasized signal
    save_audio_file("preemphasized_signal.wav", sample_rate, preemphasized_signal)

    # Plot pre-emphasized signal waveform
    plot_waveform(preemphasized_signal, "Pre-emphasized Signal")

    # Divide the signal into frames
    frame_size = 0.025  # 25 milliseconds
    frame_stride = 0.01  # 10 milliseconds
    frames = framing(preemphasized_signal, frame_size * sample_rate, frame_stride * sample_rate)

    # Save framed signal
    save_audio_file("framed_signal.wav", sample_rate, frames.flatten())

    # Plot framed signal waveform
    plot_waveform(frames.flatten(), "Framed Signal")

    # Apply Hamming window to each frame
    windowed_frames = np.apply_along_axis(hamming_window, 1, frames)

    # Save windowed signal
    save_audio_file("windowed_signal.wav", sample_rate, windowed_frames.flatten())

    # Plot windowed signal waveform
    plot_waveform(windowed_frames.flatten(), "Windowed Signal")

    # Compute MFCCs for each frame
    mfcc_features = np.apply_along_axis(compute_mfcc, 1, windowed_frames, sample_rate)

    # Simple decoding: Print the mean of MFCCs as the decoded text
    decoded_text = str(mfcc_features)
    print("Decoded Text:", decoded_text)


if __name__ == "__main__":
    audio_file_path = "/home/yasser_jemli/Signal_Processing_Project/Project/SA1.WAV"
    main(audio_file_path)
