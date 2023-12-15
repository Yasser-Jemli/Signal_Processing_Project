# Done By Yasser JEMLI 15 Dev 12:41


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import math
import contextlib
from scipy.io import wavfile
from pylab import *
import nbformat
from nbconvert import HTMLExporter

fname = '/home/celadodc-rswl.com/yasser.jamli/Signal_Processing_Project/Projects/audio_compression_2/babble_16.wav'
outname = 'filtered.wav'

cutOffFrequency = 1000.0


def fft_dis(fname):
    sampFreq, snd = wavfile.read(fname)

    snd = snd / (2.**15) #convert sound array to float pt. values

    #s1 = snd[:,0] #left channel

    #s2 = snd[:,1] #right channel

    n = len(snd)
    p = fft(snd) # take the fourier transform of left channel

    #m = len(s2) 
    #p2 = fft(s2) # take the fourier transform of right channel

    nUniquePts = int(ceil((n+1)/2.0))
    p = p[0:nUniquePts]
    p = abs(p)

    #mUniquePts = int(ceil((m+1)/2.0))
    #p2 = p2[0:mUniquePts]
    #p2 = abs(p2)
    
    p = p / float(n) # scale by the number of points so that
             # the magnitude does not depend on the length 
             # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 
# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n);
    plt.plot(freqArray/1000, 10*log10(p), color='k')
    plt.xlabel('Channel_Frequency (kHz)')
    plt.ylabel('Channel_Power (dB)')
    plt.show()

def run_mean(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)
    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

with contextlib.closing(wave.open(fname,'rb')) as spf:
    sampleRate = spf.getframerate()
    ampWidth = spf.getsampwidth()
    nChannels = spf.getnchannels()
    nFrames = spf.getnframes()

    # Extract Raw Audio from multi-channel Wav File
    signal = spf.readframes(nFrames*nChannels)
    spf.close()
    channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

    # get window size
    fqRatio = (cutOffFrequency/sampleRate)
    N = int(math.sqrt(0.196196 + fqRatio**2)/fqRatio)

    # Use moviung average (only on first channel)
    filt = run_mean(channels[0], N).astype(channels.dtype)

    wav_file = wave.open(outname, "w")
    wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
    wav_file.writeframes(filt.tobytes('C'))
    wav_file.close()
    
    
n = 0
for n in range (0,2): 
    if n==0:
        fft_dis(fname)
    elif n==1:
        fft_dis(outname)


def save_fft_plot(fname, output_filename):
    sampFreq, snd = wavfile.read(fname)

    snd = snd / (2.**15)
    n = len(snd)
    p = fft(snd)
    nUniquePts = int(ceil((n+1)/2.0))
    p = p[0:nUniquePts]
    p = abs(p)
    p = p / float(n)
    p = p**2

    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) -1] = p[1:len(p) - 1] * 2

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n)

    plt.plot(freqArray/1000, 10*log10(p), color='k')
    plt.xlabel('Channel_Frequency (kHz)')
    plt.ylabel('Channel_Power (dB)')
    plt.savefig(output_filename)
    plt.close()  # Close the plot to release resources

def generate_html_report():
    save_fft_plot(fname, 'original_signal.png')
    save_fft_plot(outname, 'filtered_signal.png')

    # Create an HTML report using nbconvert
    nb = nbformat.v4.new_notebook()

    # Markdown cell for signal details
    text = """
    # Signal Details
    <h2>Original Signal</h2>
    <img src="original_signal.png" alt="Original Signal" style="width:50%">
    
    <h2>Filtered Signal</h2>
    <img src="filtered_signal.png" alt="Filtered Signal" style="width:50%">
    
    Add more signal details here...
    """
    nb['cells'] = [nbformat.v4.new_markdown_cell(text)]

    # Convert to HTML
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb, resources={'output_files_dir': 'output_images'})

    # Write HTML report to a file
    with open('signal_report.html', 'w') as report_file:
        report_file.write(body)

    # Move generated images to the output_images directory
    image_files = resources.get('outputs', {}).get('text/html', [])
    for file in image_files:
        os.rename(file, os.path.join('output_images', os.path.basename(file)))

# Run the functions to generate the HTML report
generate_html_report()