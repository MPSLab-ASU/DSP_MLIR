#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100000000

// Function to generate a range of values
void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
}

// Function to apply gain (multiplier) to a signal
void gain(double* output, double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

// Function to compute the sine of each element in the input array
void sine(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = sin(input[i]);
    }
}

// Function to add two signals element-wise
void add(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

// Function to delay the signal by a certain number of samples
void delay(double* input, double* output, int delaySamples, int length) {
    for (int i = 0; i < length; i++) {
        if (i < delaySamples) {
            output[i] = 0; // Initial delay period is zeroed
        } else {
            output[i] = input[i - delaySamples];
        }
    }
}

// LMS filter response function
void lmsFilterResponse(double* output, double* noisy_sig, double* clean_sig, double mu, int filterSize, int length) {
    double w[32] = {0}; // Initialize weights to zero
    for (int n = 0; n < length; n++) {
        double y = 0;
        for (int i = 0; i < filterSize; i++) {
            if (n - i >= 0) {
                y += w[i] * noisy_sig[n - i];
            }
        }
        double e = clean_sig[n] - y;
        for (int i = 0; i < filterSize; i++) {
            if (n - i >= 0) {
                w[i] += mu * e * noisy_sig[n - i];
            }
        }
        output[n] = y;
    }
}

int main() {
    int fs = 8000;
    double step = 1.0 / fs;

    // Allocate memory for vectors
    double* input = (double*)malloc(INPUT_LENGTH * sizeof(double));
    double* getSinDuration = (double*)malloc(INPUT_LENGTH * sizeof(double));
    double* clean_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    double* noise = (double*)malloc(INPUT_LENGTH * sizeof(double));
    double* noisy_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    double* y = (double*)malloc(INPUT_LENGTH * sizeof(double));

    // Generate input range
    getRangeOfVector(input, 0.0, INPUT_LENGTH, step);

    // Generate clean signal
    double f_sig = 500;
    gain(getSinDuration, input, 2 * PI * f_sig, INPUT_LENGTH);
    
    sine(clean_sig, getSinDuration, INPUT_LENGTH);

    // Generate noise signal with a delay of 2 samples
    delay(clean_sig, noise, 2, INPUT_LENGTH);

    // Create noisy signal by adding noise to clean signal
    add(noisy_sig, clean_sig, noise, INPUT_LENGTH);

    // Apply LMS filter
    double mu = 0.01;
    int filterSize = 32;

    lmsFilterResponse(y, noisy_sig, clean_sig, mu, filterSize, INPUT_LENGTH);

    // Print result (for demonstration purposes)
    for (int i = 0; i < INPUT_LENGTH && i < 10; i++) { // Limit print to first few samples
        printf("%f\n", y[i]);
    }

    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(clean_sig);
    free(noise);
    free(noisy_sig);
    free(y);

    return 0;
}