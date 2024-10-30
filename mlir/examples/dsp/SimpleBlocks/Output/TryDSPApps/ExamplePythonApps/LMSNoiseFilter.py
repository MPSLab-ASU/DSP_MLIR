import numpy as np
import matplotlib.pyplot as plt

# Sample Input Signal: A noisy sinusoidal signal
np.random.seed(42)
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
clean_signal = 0.5 * np.sin(2 * np.pi * 50 * t)  # Clean 50 Hz sine wave
noise = np.random.normal(0, 0.1, clean_signal.shape)
input_signal = clean_signal + noise

# DSP Blocks Implementation

def low_pass_filter(signal, cutoff_freq, fs):
    # Simple RC low-pass filter
    alpha = 1 / (2 * np.pi * cutoff_freq / fs + 1)
    filtered_signal = np.zeros_like(signal)
    for i in range(1, len(signal)):
        filtered_signal[i] = alpha * signal[i] + (1 - alpha) * filtered_signal[i-1]
    return filtered_signal

def downsample(signal, factor):
    return signal[::factor]

def lms_filter(signal, reference_noise, mu=0.01, order=4):
    N = len(signal)
    weights = np.zeros(order)
    output = np.zeros(N)
    for n in range(order, N):
        x = signal[n-order:n][::-1]
        y = np.dot(weights, x)
        error = signal[n] - y
        weights += 2 * mu * error * x
        output[n] = y
    return output

def gain(signal, gain_factor):
    return signal * gain_factor

def upsample(signal, factor):
    return np.repeat(signal, factor)

def high_pass_filter(signal, cutoff_freq, fs):
    # Simple RC high-pass filter
    alpha = 1 / (2 * np.pi * cutoff_freq / fs + 1)
    filtered_signal = np.zeros_like(signal)
    for i in range(1, len(signal)):
        filtered_signal[i] = alpha * (filtered_signal[i-1] + signal[i] - signal[i-1])
    return filtered_signal

# Applying the DSP Blocks in Sequence

# Step 2: LowPassFilter
cutoff_freq_low = 100  # Low-pass filter cutoff frequency
filtered_signal = low_pass_filter(input_signal, cutoff_freq_low, fs)

# Step 3: Downsampling
downsample_factor = 2
downsampled_signal = downsample(filtered_signal, downsample_factor)

# Step 4: LMS Filter
reference_noise = np.random.normal(0, 0.1, downsampled_signal.shape)  # Simulated reference noise
cleaned_signal = lms_filter(downsampled_signal, reference_noise)

# Step 5: Gain
gain_factor = 2
amplified_signal = gain(cleaned_signal, gain_factor)

# Step 6: Upsampling
upsample_factor = downsample_factor
upsampled_signal = upsample(amplified_signal, upsample_factor)

# Step 7: HighPassFilter
cutoff_freq_high = 5  # High-pass filter cutoff frequency
final_signal = high_pass_filter(upsampled_signal, cutoff_freq_high, fs)

# Plotting the results

plt.figure(figsize=(15, 10))

plt.subplot(5, 1, 1)
plt.plot(t, input_signal)
plt.title('Input Signal')
plt.grid()

plt.subplot(5, 1, 2)
plt.plot(t, filtered_signal)
plt.title('After Low Pass Filter')
plt.grid()

plt.subplot(5, 1, 3)
plt.plot(t[:len(downsampled_signal)], downsampled_signal)
plt.title('After Downsampling')
plt.grid()

plt.subplot(5, 1, 4)
plt.plot(t[:len(cleaned_signal)], cleaned_signal)
plt.title('After LMS Filter')
plt.grid()

plt.subplot(5, 1, 5)
plt.plot(t, final_signal)
plt.title('Final Output Signal')
plt.grid()

plt.tight_layout()
plt.show()
