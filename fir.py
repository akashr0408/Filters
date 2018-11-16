import  numpy as np
import scipy.signal as sp
from scipy.fftpack import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt

sampling_rate, input_signal = wavfile.read("440_1320_chirp_plus_800_1200_square.wav", 'r')
number_frames = len(input_signal)

nyquist_rate = sampling_rate / 2.0
width = 5.0 / nyquist_rate
attenuation = 60 

#kaiser parameter calculation
N, beta = sp.kaiserord(attenuation, width)

cutoff_freq = 1320
taps = sp.firwin(N, cutoff_freq/nyquist_rate, window=('kaiser', beta))
filtered_signal = sp.lfilter(taps, 1.0, input_signal)

#perfrom fft on both signals to comapre them

in_sig_fft = fft(input_signal)
filtered_sig_fft = fft(filtered_signal)

in_sig_fft = np.abs(in_sig_fft)
filtered_sig_fft = np.abs(filtered_sig_fft)

if len(in_sig_fft)%2 > 0:
    in_sig_fft[1:len(in_sig_fft)] = in_sig_fft[1:len(in_sig_fft)]*2 
else:
    in_sig_fft[1:len(in_sig_fft)-1] = in_sig_fft[1:len(in_sig_fft) - 1]*2

if len(filtered_sig_fft)%2 > 0:
    filtered_sig_fft[1:len(filtered_sig_fft)] = filtered_sig_fft[1:len(filtered_sig_fft)]*2 
else:
    filtered_sig_fft[1:len(filtered_sig_fft)-1] = filtered_sig_fft[1:len(filtered_sig_fft) - 1]*2

#convert to decibel
db_in_fft = 10 * np.log10(in_sig_fft)
db_filtered_fft = 10 * np.log10(filtered_sig_fft)

#plot signals

freq = np.arange(len(input_signal)) * sampling_rate / len(input_signal)

plt.figure(1)
plt.plot(freq, db_in_fft, 'r')
plt.plot(freq, db_filtered_fft, 'b')
plt.title('Original Input Signal (red) vs FIR filtered signal (blue)')

plt.grid(True)
plt.show()
