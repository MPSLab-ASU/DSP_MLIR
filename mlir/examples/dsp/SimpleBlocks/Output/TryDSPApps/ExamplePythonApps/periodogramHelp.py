import numpy as np

# Define the functions for Fourier Transform and Convolution
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def convolve(x, h):
    N = len(x)
    M = len(h)
    y_len = N + M - 1
    y = np.zeros(y_len)
    for n in range(y_len):
        for k in range(N):
            if 0 <= n - k < M:
                y[n] += x[k] * h[n - k]
    return y

# Define the signal x[l]
x = np.array([1, 2])

# Create the time-reversed signal x[-l]
x_neg = x[::-1]
print(x_neg)
# Compute the convolution of x[l] and x[-l]
conv_result = convolve(x, x_neg)
# conv_result = convolve(x, x_neg)

# Compute the Fourier Transform of the convolution result
conv_fft = dft(conv_result)
x1 = np.array([1, 2,3,2, 1])
print(dft(x1))
# Zero-pad x and x_neg to the length of the convolution result
# padded_len = len(conv_result)
# x_padded = np.pad(x, (0, padded_len - len(x)))
# x_neg_padded = np.pad(x_neg, (0, padded_len - len(x_neg)))
# # print(x_padded)
# # print(x_neg_padded)

# # Compute the Fourier Transform of padded x[l] and x[-l]
# X_w = dft(x_padded)
# X_neg_w = dft(x_neg_padded)

# print("Printing X_w & X_neg_w")
# print(X_w)
# print(X_neg_w)
# Compute the product of the Fourier Transforms
# product_fft = X_w * X_neg_w
# prod1_fft = dft(x) * dft(x_neg)

# Print the results
print("Convolution result: ", conv_result)
print("Fourier Transform of convolution: ", conv_fft)
# print("Product of individual Fourier Transforms: ", product_fft)
# print("Product of individual Fourier Transforms No Padding: ", prod1_fft)

# Verify the results
# print("Verification (are they equal?): ", np.allclose(conv_fft, product_fft))
# print("Verification (are they equal?): ", np.allclose(conv_fft, product_fft))
