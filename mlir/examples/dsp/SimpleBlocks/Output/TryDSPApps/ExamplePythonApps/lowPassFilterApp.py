import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter

# Signal parameters
fs = 8000  # Sampling rate (Hz)
# t = np.arange(0, 0.05, 1.0/fs)  # Time vector (50 milliseconds for better visualization)
# print(t) 0.001125
t = np.arange(0, 0.006375, 1.0/fs)
print(len(t))
# print(t)
# t1 = np.arange(0, 10, 2)
# print(t1)
print(1.0/fs)
# Generate a clean sine wave (signal frequency 500 Hz)
f_signal = 500  # Signal frequency (Hz)
clean_signal = np.sin(2 * np.pi * f_signal * t)

# Generate high-frequency noise (frequency 3000 Hz)
f_noise = 3000  # Noise frequency (Hz)
noise = 0.5 * np.sin(2 * np.pi * f_noise * t)

# Add noise to the clean signal
noisy_signal = clean_signal + noise
# print(noisy_signal)
# Design a low-pass FIR filter
N = 51  # Filter length (must be odd for symmetry)
cutoff_freq = 1000  # Cutoff frequency (Hz)
wc = 2 * np.pi * cutoff_freq / fs  # Normalized cutoff frequency

# Define the ideal low-pass filter impulse response
def ideal_lp(n, wc):
    if n == 0:
        return wc / np.pi
    else:
        return np.sin(wc * n) / (np.pi * n)

# Create the low-pass filter coefficients
h_lp = np.array([ideal_lp(n - (N-1)//2, wc) for n in range(N)])
# print(h_lp)
# Apply a Hamming window to the low-pass filter coefficients
hamming_window = np.hamming(N)
# print(np.hamming(5))
h_lp_w = h_lp * hamming_window
print(h_lp_w)
# Apply the low-pass filter to the noisy signal
filtered_signal = lfilter(h_lp_w, 1.0, noisy_signal)
print(filtered_signal)
print(len(filtered_signal))
exit()
# Plot the original, noisy, and filtered signals
plt.figure(figsize=(12, 8))

# Original clean signal
plt.subplot(3, 1, 1)
plt.plot(t, clean_signal, label='Clean Signal')
plt.title('Clean Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

# Noisy signal
plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.title('Noisy Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

# Filtered signal
plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.title('Filtered Signal (After Low-Pass Filter)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

# Plot the frequency response of the filter
w, h = freqz(h_lp_w, worN=8000)
plt.figure(figsize=(8, 4))
plt.plot(w / np.pi * (fs / 2), np.abs(h), 'b')
plt.title('Low-Pass Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()