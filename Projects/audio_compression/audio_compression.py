import numpy as np
import scipy.io.wavfile as wav
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import scipy.stats

def read_audio_file(file_path):
    rate, data = wav.read(file_path)
    return rate, data

def save_audio_file(file_path, rate, data):
    wav.write(file_path, rate, data)

def quantize(signal, num_bits):
    quantized_signal = np.round(signal / (2**(16 - num_bits)))
    return quantized_signal.astype(np.int16)

def huffman_coding(data):
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
    compressed_data = ''.join([huffman_dict[symbol] for symbol in data])
    return compressed_data

def decompress_audio(compressed_data, huffman_dict):
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
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig(save_path)
    plt.show(block=False)  # Show the plot without blocking
    plt.pause(3)  # Pause for 3 seconds
    plt.close()

def analyze_original_signal(audio_data):
    plt.hist(audio_data, bins='auto')
    plt.title("Histogram of Original Signal Values")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    mean_value = np.mean(audio_data)
    std_dev = np.std(audio_data)
    skewness = scipy.stats.skew(audio_data)

    print(f"Mean: {mean_value}, Standard Deviation: {std_dev}, Skewness: {skewness}")

    # Assuming the original bit depth is 16 bits
    original_bit_depth = 16
    print(f"Original Bit Depth: {original_bit_depth} bits")

def main(file_path, num_bits):
    rate, audio_data = read_audio_file(file_path)

    analyze_original_signal(audio_data)

    plot_signal(audio_data, "Original Audio Signal", "original_signal.png")

    quantized_data = quantize(audio_data, num_bits)

    plot_signal(quantized_data, "Quantized Audio Signal", "quantized_signal.png")

    huffman_dict = huffman_coding(quantized_data)
    compressed_data = compress_audio(quantized_data, huffman_dict)

    plot_signal([int(bit) for bit in compressed_data], "Compressed Audio Signal (Bits)", "compressed_bits.png")

    decompressed_data = decompress_audio(compressed_data, huffman_dict)

    plot_signal(decompressed_data, "Decompressed Audio Signal", "decompressed_signal.png")

    save_audio_file("compressed_audio.wav", rate, decompressed_data)

if __name__ == "__main__":
    audio_file_path = "/home/yasser_jemli/Signal_Processing_Project/Projects/audio_compression/SA1.WAV"
    num_bits = 4
    main(audio_file_path, num_bits)
