#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 10
#define FILTER_LENGTH 32

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

// Corrected LMS filter response function
void lmsFilterResponse(double* y, double* noisy_sig, double* clean_sig, double mu, int filterSize, int length) {
    double w[FILTER_LENGTH] = {0}; // Initialize weights to zero
    for (int n = 0; n < length; n++) {
        y[n] = 0;
        for (int i = 0; i < filterSize; i++) {
            if (n - i >= 0) {
                y[n] += w[i] * noisy_sig[n - i];
            }
        }
        double e = clean_sig[n] - y[n];
        for (int i = 0; i < filterSize; i++) {
            if (n - i >= 0) {
                w[i] += mu * e * noisy_sig[n - i];
            }
        }
        
    }
}

int main() {
    int fs = 8000;
    double step = 1.0 / fs;

    // Allocate memory for vectors
    double input[INPUT_LENGTH];
    double getSinDuration[INPUT_LENGTH];
    double clean_sig[INPUT_LENGTH];
    double getNoiseSinDuration[INPUT_LENGTH];
    double noise[INPUT_LENGTH];
    double noise1[INPUT_LENGTH];
    double noisy_sig[INPUT_LENGTH];
    double y[INPUT_LENGTH];
    double sol[INPUT_LENGTH];

    // Generate input range
    getRangeOfVector(input, 0.0, INPUT_LENGTH, step);

    // Generate clean signal
    double f_sig = 500;
    gain(getSinDuration, input, 2 * PI * f_sig, INPUT_LENGTH);
    sine(clean_sig, getSinDuration, INPUT_LENGTH);

    // Generate noise signal with frequency of 3000 Hz
    double f_noise = 3000;
    gain(getNoiseSinDuration, input, 2 * PI * f_noise, INPUT_LENGTH);
    sine(noise, getNoiseSinDuration, INPUT_LENGTH);


    gain(noise1, noise, 0.5, INPUT_LENGTH);

    // Create noisy signal by adding noise to clean signal
    add(noisy_sig, clean_sig, noise1, INPUT_LENGTH);


    double mu = 0.01;
    lmsFilterResponse(y, noisy_sig, clean_sig, mu, FILTER_LENGTH, INPUT_LENGTH);


    double G1 = 123;
    gain(sol, y, G1, INPUT_LENGTH);

   
    printf("%f\n", sol[3]);


    return 0;
}