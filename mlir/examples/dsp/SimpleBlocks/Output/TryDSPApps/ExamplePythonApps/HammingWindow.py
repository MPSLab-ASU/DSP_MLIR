import numpy as np
import matplotlib.pyplot as plt

def hamming_window(N):
    n = np.arange(N)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    return window

# Example usage
N = 5  # Number of points in the window
window = hamming_window(N)
print(window)
# Plot the Hamming window
plt.figure(figsize=(10, 6))
plt.plot(window, label='Hamming Window')
plt.title('Symmetric Hamming Window')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
