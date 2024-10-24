import numpy as np

def quantize(coefficients, num_bits):
    min_value = -5
    max_value = 5
    num_levels = 2**num_bits
    step_size = (max_value - min_value) / num_levels
    
    # Quantization process
    quantized_coefficients = np.round((coefficients - min_value) / step_size) * step_size + min_value
    
    return quantized_coefficients

# Example input vector
X_thresh = np.array([3.2, -1.5, 0.8, -2.9, 4.5])

# Quantize with 4 bits
num_bits = 4
X_quant = quantize(X_thresh, num_bits)

print("Original coefficients:", X_thresh)
print("Quantized coefficients:", X_quant)
