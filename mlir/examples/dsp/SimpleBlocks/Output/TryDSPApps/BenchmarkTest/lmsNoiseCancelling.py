import numpy as np
import matplotlib.pyplot as plt

# Define constants
N = 100        # Length of signal
M = 32        # Number of filter coefficients
MU = 0.01       # Step size (learning rate)

fs = 8000  # Sampling rate (Hz)
# t = np.arange(0, 0.05, 1.0/fs)  # Time vector (50 milliseconds for better visualization)
# print(t) 0.001125
t = np.arange(0, 0.025, 1.0/fs)
# print(len(t))
# print(t)
# t1 = np.arange(0, 10, 2)
# print(t1)
# print(1.0/fs)
# Generate a clean sine wave (signal frequency 500 Hz)
f_signal = 500  # Signal frequency (Hz)
clean_signal = np.sin(2 * np.pi * f_signal * t)

# Generate high-frequency noise (frequency 3000 Hz)
f_noise = 3000  # Noise frequency (Hz)
noise = 0.5 * np.sin(2 * np.pi * f_noise * t)

# Add noise to the clean signal
noisy_signal = clean_signal + noise

# print(noisy_signal)


def lms_filter(x, d):
    # Initialize variables
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    
    # LMS filter algorithm
    for n in range(N):
        # Calculate the filter output y[n]
        y[n] = sum(w[i] * x[n - i] for i in range(M) if n - i >= 0)
        
        # Calculate the error e[n]
        e[n] = d[n] - y[n]
        
        # Update the filter weights w[i]
        for i in range(M):
            if n - i >= 0:
                w[i] += MU * e[n] * x[n - i]
    
    return y, e, w

# Apply LMS filter
y, e, w = lms_filter( noisy_signal, clean_signal)

print(y)

# Plot the signals
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(clean_signal)
plt.title('Desired Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(noisy_signal)
plt.title('Input Signal (with noise)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(y)
plt.title('Output Signal (filtered)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
