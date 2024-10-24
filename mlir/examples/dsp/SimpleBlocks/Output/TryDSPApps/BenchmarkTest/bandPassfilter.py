import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz

# Signal parameters
fs = 8000  # Sampling rate (Hz)
t = np.arange(0, 0.05, 1.0/fs)  # Time vector (1 second)

# Generate a signal with multiple frequency components
f1 = 500  # Frequency component 1 (Hz)
f2 = 1500  # Frequency component 2 (Hz)
f3 = 2500  # Frequency component 3 (Hz)
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)

# Design parameters for band-pass filter
N = 51  # Filter length (must be odd for symmetry)
low_cutoff = 1000  # Lower cutoff frequency (Hz)
high_cutoff = 2000  # Upper cutoff frequency (Hz)

# Normalized cutoff frequencies
wc1 = 2 * np.pi * low_cutoff / fs
wc2 = 2 * np.pi * high_cutoff / fs

# Define the ideal low-pass filter impulse response
def ideal_lp(n, wc):
    if n == 0:
        return wc / np.pi
    else:
        return np.sin(wc * n) / (np.pi * n)

# Create the low-pass filter coefficients for both cutoffs
h_lp1 = np.array([ideal_lp(n - (N-1)//2, wc1) for n in range(N)])
h_lp2 = np.array([ideal_lp(n - (N-1)//2, wc2) for n in range(N)])

# Apply a Hamming window to the low-pass filter coefficients
hamming_window = np.hamming(N)
h_lp1_w = h_lp1 * hamming_window
h_lp2_w = h_lp2 * hamming_window

# Construct the band-pass filter by subtracting the two low-pass filters
h_bp = h_lp2_w - h_lp1_w

# Apply the band-pass filter to the signal
filtered_signal = lfilter(h_bp, 1.0, signal)

# Plot the original and filtered signals
plt.figure(figsize=(12, 8))

# Original signal
plt.subplot(3, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

# Filtered signal
plt.subplot(3, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Signal (Band-Pass)')
plt.title('Filtered Signal (After Band-Pass Filter)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

# Plot the frequency response of the band-pass filter
w, h = freqz(h_bp, worN=8000)
plt.figure(figsize=(8, 4))
plt.plot(w / np.pi * (fs / 2), np.abs(h), 'b')
plt.title('Band-Pass Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()
