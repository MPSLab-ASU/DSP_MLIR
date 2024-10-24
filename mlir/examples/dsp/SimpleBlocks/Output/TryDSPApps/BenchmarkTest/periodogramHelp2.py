import numpy as np

# Define the functions for Fourier Transform and Convolution
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

def convolve(x, h):
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        for k in range(N):
            y[n] += x[k] * h[(n - k) % N]
    return y

# Define the signal x[l]
x = np.array([1, 2])

# Create the time-reversed signal x[-l]
x_neg = np.array([x[-i] for i in range(len(x))])

# Compute the convolution of x[l] and x[-l]
conv_result = convolve(x, x_neg)

# Compute the Fourier Transform of the convolution result
conv_fft = dft(conv_result)

# Compute the Fourier Transform of x[l] and x[-l]
X_w = dft(x)
X_neg_w = dft(x_neg)

# Compute the product of the Fourier Transforms
product_fft = X_w * X_neg_w

# Print the results
print("Convolution result: ", conv_result)
print("Fourier Transform of convolution: ", conv_fft)
# print("Product of individual Fourier Transforms: ", product_fft)

# Verify the results
print("Verification (are they equal?): ", np.allclose(conv_fft, product_fft))
