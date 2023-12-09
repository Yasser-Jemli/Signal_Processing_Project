import numpy as np
import scipy.io.wavfile as wav
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt

def read_audio_file(file_path):
    # Read the audio file
    rate, data = wav.read(file_path)
    return rate, data

def save_audio_file(file_path, rate, data):
    # Save the compressed audio file
    wav.write(file_path, rate, data)

def quantize(signal, num_bits):
    # Quantize the signal using a specified number of bits
    quantized_signal = np.round(signal / (2**(16 - num_bits)))
    return quantized_signal.astype(np.int16)

def huffman_coding(data):
    # Implement Huffman coding
    frequencies = defaultdict(int)
    for symbol in data:
        frequencies[symbol] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_dict = dict(heap[0][1:])
    return huffman_dict

def compress_audio(data, huffman_dict):
    # Compress the data using Huffman coding
    compressed_data = ''.join([huffman_dict[symbol] for symbol in data])
    return compressed_data

def decompress_audio(compressed_data, huffman_dict):
    # Decompress the data using Huffman coding
    current_code = ''
    decompressed_data = []
    for bit in compressed_data:
        current_code += bit
        for symbol, code in huffman_dict.items():
            if current_code == code:
                decompressed_data.append(symbol)
                current_code = ''
                break
    return np.array(decompressed_data, dtype=np.int16)

def plot_signal(signal, title, save_path):
    # Plot and save the signal
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig(save_path)
    plt.close()

def main(file_path, num_bits):
    # Read the audio file
    rate, audio_data = read_audio_file(file_path)

    # Plot the original audio signal
    plot_signal(audio_data, "Original Audio Signal", "original_signal.png")

    # Quantize the audio data
    quantized_data = quantize(audio_data, num_bits)

    # Plot the quantized signal
    plot_signal(quantized_data, "Quantized Audio Signal", "quantized_signal.png")

    # Huffman coding
    huffman_dict = huffman_coding(quantized_data)
    compressed_data = compress_audio(quantized_data, huffman_dict)

    # Plot the compressed signal (bits)
    plot_signal([int(bit) for bit in compressed_data], "Compressed Audio Signal (Bits)", "compressed_bits.png")

    # Decompression
    decompressed_data = decompress_audio(compressed_data, huffman_dict)

    # Plot the decompressed signal
    plot_signal(decompressed_data, "Decompressed Audio Signal", "decompressed_signal.png")

    # Save the decompressed audio file
    save_audio_file("compressed_audio.wav", rate, decompressed_data)

if __name__ == "__main__":
    audio_file_path = "your_audio_file.wav"
    num_bits = 4  # Choose the number of bits for quantization
    main(audio_file_path, num_bits)
