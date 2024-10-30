import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz

# Function to generate a sample input signal (sum of sine waves)
def generate_input_signal(duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t)
    return signal, t

# Function to design and apply an FIR filter
def apply_fir_filter(signal, cutoff, filter_type, sample_rate=44100, numtaps=101):
    nyquist = 0.5 * sample_rate
    normalized_cutoff = np.array(cutoff) / nyquist
    taps = firwin(numtaps, normalized_cutoff, pass_zero=(filter_type == 'low' or filter_type == 'bandstop'))
    filtered_signal = lfilter(taps, 1.0, signal)
    return filtered_signal, taps

# Function to apply gain to a signal
def apply_gain(signal, gain):
    return gain * signal

# Function to plot the frequency response of the filters
def plot_frequency_response(taps, sample_rate=44100):
    w, h = freqz(taps, worN=8000)
    plt.plot(0.5 * sample_rate * w / np.pi, np.abs(h), 'b')
    plt.title("Frequency Response")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid()

# Main function to demonstrate the audio equalizer
def main():
    sample_rate = 44100
    duration = 1.0
    input_signal, t = generate_input_signal(duration, sample_rate)
    print(len(t))
    print(len(input_signal))
    t = t[:3000]
    input_signal = input_signal[:3000]

    # Apply FIR filters
    low_pass_signal, low_pass_taps = apply_fir_filter(input_signal, cutoff=300, filter_type='low', sample_rate=sample_rate)
    high_pass_signal, high_pass_taps = apply_fir_filter(input_signal, cutoff=1000, filter_type='high', sample_rate=sample_rate)
    band_pass_signal, band_pass_taps = apply_fir_filter(input_signal, cutoff=[300, 1000], filter_type='bandpass', sample_rate=sample_rate)

    # Apply gain to each filtered signal
    low_pass_signal = apply_gain(low_pass_signal, gain=0.5)
    high_pass_signal = apply_gain(high_pass_signal, gain=0.5)
    band_pass_signal = apply_gain(band_pass_signal, gain=1.0)

    # Summation of the filtered signals
    equalized_signal = low_pass_signal + high_pass_signal + band_pass_signal

    # Plot the input and output signals on a proper scale
    plt.figure(figsize=(15, 12))
    plt.subplot(4, 1, 1)
    plt.plot(t, input_signal, label='Input Signal')
    plt.title('Input Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, equalized_signal, label='Equalized Signal')
    plt.title('Equalized Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, input_signal - equalized_signal, label='Difference Signal')
    plt.title('Difference Signal (Input - Equalized)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 4)
    plot_frequency_response(low_pass_taps, sample_rate)
    plot_frequency_response(high_pass_taps, sample_rate)
    plot_frequency_response(band_pass_taps, sample_rate)
    plt.legend(['Low-pass Filter', 'High-pass Filter', 'Band-pass Filter'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
