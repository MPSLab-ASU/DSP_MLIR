import numpy as np
import matplotlib.pyplot as plt

# Assuming these functions are defined in your environment
def lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize):
    # Simple LMS filter implementation (for demonstration purposes)
    w = np.zeros(filterSize)
    output = np.zeros(len(noisy_sig))
    for n in range(len(noisy_sig) - filterSize):
        x = noisy_sig[n:n + filterSize]
        d = clean_sig[n + filterSize]
        y = np.dot(w, x)
        e = d - y
        w = w + 2 * mu * e * x
        output[n + filterSize] = y
    return output

def amplification(input_audio, gain):
    return gain * input_audio

def spectral_subtraction(noisy_sig, noise_estimate):
    return noisy_sig - noise_estimate

def feedback_suppression(input_audio, clean_audio, mu, filter_size):
    return lmsFilterResponse(input_audio, clean_audio, mu, filter_size)

# Parameters
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
freq = 5  # Frequency of the input signal

# Generating a sinusoidal input signal
input_audio = np.sin(2 * np.pi * freq * t)
gain = 2
noise_estimate = 0.5 * np.random.randn(len(input_audio))  # Random noise estimate
mu = 0.01
filter_size = 5
clean_audio = input_audio  # Clean audio is the original sine wave

# Processing steps
amplified_audio = amplification(input_audio, gain)
noisy_sig = amplified_audio + noise_estimate  # Simulating a noisy signal
noise_reduced_audio = spectral_subtraction(noisy_sig, noise_estimate)
enhanced_audio = feedback_suppression(noise_reduced_audio, clean_audio, mu, filter_size)

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, input_audio)
plt.title('Input Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(t, amplified_audio)
plt.title('Amplified Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.plot(t, noise_reduced_audio)
plt.title('Noise Reduced Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 4)
plt.plot(t, enhanced_audio)
plt.title('Enhanced Audio')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
