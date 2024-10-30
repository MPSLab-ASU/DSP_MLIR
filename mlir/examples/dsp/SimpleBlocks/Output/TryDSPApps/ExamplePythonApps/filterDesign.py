import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Define the parameters
N = 51  # Length of the filter (choose an odd number for symmetry)
wc = np.pi / 4  # Cutoff frequency (normalized, 0 to pi)

# Define the ideal low-pass filter impulse response
def ideal_lp(n, wc):
    if n == 0:
        return wc / np.pi
    else:
        return np.sin(wc * n) / (np.pi * n)

# Create the low-pass filter coefficients
h_lp = np.array([ideal_lp(n - (N-1)//2, wc) for n in range(N)])

# Define the Hamming window
hamming_window = np.hamming(N)

# Apply the window to the low-pass filter coefficients
h_lp_w = h_lp * hamming_window

# Perform spectral inversion to get high-pass filter coefficients
h_hp = np.zeros(N)
h_hp[(N-1)//2] = 1 - h_lp_w[(N-1)//2]  # Center coefficient
h_hp[:(N-1)//2] = -h_lp_w[:(N-1)//2]    # Negative coefficients
h_hp[(N-1)//2+1:] = -h_lp_w[(N-1)//2+1:]  # Negative coefficients

# Calculate frequency response of the low-pass filter
w_lp, H_lp = freqz(h_lp_w, worN=8000)
# Calculate frequency response of the high-pass filter
w_hp, H_hp = freqz(h_hp, worN=8000)

# Plot the frequency response
plt.figure(figsize=(14, 6))

# Low-pass filter frequency response
plt.subplot(1, 2, 1)
plt.plot(w_lp / np.pi, np.abs(H_lp), 'b')
plt.title('Low-Pass Filter Frequency Response')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude')
plt.grid()

# High-pass filter frequency response
plt.subplot(1, 2, 2)
plt.plot(w_hp / np.pi, np.abs(H_hp), 'r')
plt.title('High-Pass Filter Frequency Response')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude')
plt.grid()

plt.tight_layout()
plt.show()