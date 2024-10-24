import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

def spectral_subtraction(input_signal, noise_estimation, alpha=1.0, beta=0.002):
    # Perform FFT on the input signal and noise estimation
    input_spectrum = fft(input_signal)
    noise_spectrum = fft(noise_estimation)
    
    # Estimate the noise power spectrum
    noise_power_spectrum = np.abs(noise_spectrum)**2
    
    # Estimate the input power spectrum
    input_power_spectrum = np.abs(input_spectrum)**2
    
    # Subtract the noise power spectrum from the input power spectrum
    cleaned_power_spectrum = input_power_spectrum - alpha * noise_power_spectrum
    
    # Apply a noise floor to avoid negative values
    cleaned_power_spectrum = np.maximum(cleaned_power_spectrum, beta * noise_power_spectrum)
    
    # Combine the phase of the input signal with the cleaned magnitude spectrum
    cleaned_spectrum = np.sqrt(cleaned_power_spectrum) * np.exp(1j * np.angle(input_spectrum))
    
    # Perform inverse FFT to get the time-domain signal
    cleaned_signal = ifft(cleaned_spectrum)
    
    # Return the real part of the cleaned signal
    return np.real(cleaned_signal)

# Generate a sample input signal (e.g., a sine wave) and noise
fs = 500  # Sampling frequency
t = np.linspace(0, 1, fs)  # Time vector
clean_signal = 0.5 * np.sin(2 * np.pi * 5 * t)  # Clean signal (5 Hz sine wave)
noise = 0.2 * np.random.normal(size=fs)  # White noise
noisy_signal = clean_signal + noise  # Noisy signal

# Use a portion of the noisy signal to estimate noise
# Here, we simply take the first 100 samples and average to match the signal length
noise_estimation = noisy_signal[:100]
noise_estimation = np.tile(noise_estimation, int(np.ceil(len(noisy_signal) / len(noise_estimation))))[:len(noisy_signal)]

# Apply noise reduction
cleaned_signal = spectral_subtraction(noisy_signal, noise_estimation)

# Plot the signals
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(t, clean_signal)
plt.title('Clean Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, cleaned_signal)
plt.title('Cleaned Signal (After Noise Reduction)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
