#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define SAMPLE_RATE 1000
#define INPUT_LENGTH 100
#define THRESHOLD 0.01

// Function prototypes
double* getRangeOfVector(double start, int length, double increment);
void gain(double* output, const double* input, double gainFactor, int length);
void sine(double* output, const double* input, int length);
void delay(double* output, const double* input, int length, int delay_steps);
void add(double* output, const double* input1, const double* input2, int length);
void threshold(double* output, const double* input, double threshold);
double zeroCrossCount(const double* input, int length);
void printArray(const double* array, int length);

// Generate a range of values
double* getRangeOfVector(double start, int length, double increment) {
    double* vector = malloc(length * sizeof(double));
    if (!vector) {
        perror("Memory allocation failed in getRangeOfVector");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
    return vector;
}

// Apply gain to a signal
void gain(double* output, const double* input, double gainFactor, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * gainFactor;
    }
}

// Compute sine wave signal
void sine(double* output, const double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = sin(input[i]);
    }
}

// Apply delay to a signal
void delay(double* output, const double* input, int length, int delay_steps) {
    for (int i = 0; i < length; i++) {
        if (i >= delay_steps) {
            output[i] = input[i - delay_steps];
        } else {
            output[i] = 0.0;
        }
    }
}

// Perform element-wise addition
void add(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

// Apply thresholding
void threshold(double* output, const double* input, double threshold) {
    for (int i = 0; i < INPUT_LENGTH; i++) {
        if (input[i] >= -threshold && input[i] <= threshold) {
            output[i] = 0.0;
        } else {
            output[i] = input[i];
        }
    }
}

// Count zero crossings and return as double
double zeroCrossCount(const double* input, int length) {
    double count = 0.0;
    for (int i = 1; i < length; i++) {
        if ((input[i - 1] > 0 && input[i] < 0) || (input[i - 1] < 0 && input[i] > 0)) {
            count += 1.0;  // Increment as double
        }
    }
    return count;
}

// Print an array
void printArray(const double* array, int length) {
    for (int i = 0; i < length; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

int main() {
    // Step 1: Generate Input Vector
    double* input = getRangeOfVector(0, INPUT_LENGTH, 0.0125);

    // Step 2: Apply Gain
    double* getSinDuration = malloc(INPUT_LENGTH * sizeof(double));
    gain(getSinDuration, input, 2 * PI * 5, INPUT_LENGTH);

    // Step 3: Compute Sine Wave Signal
    double* signal = malloc(INPUT_LENGTH * sizeof(double));
    sine(signal, getSinDuration, INPUT_LENGTH);

    // Step 4: Generate Delayed Noise Signal
    double* noise = malloc(INPUT_LENGTH * sizeof(double));
    delay(noise, signal, INPUT_LENGTH, 5);

    // Step 5: Compute Noisy Signal
    double* noisy_sig = malloc(INPUT_LENGTH * sizeof(double));
    add(noisy_sig, signal, noise, INPUT_LENGTH);

    // Step 6: Apply Thresholding
    double* GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));
    threshold(GetThresholdReal, noisy_sig, THRESHOLD);

    // Step 7: Compute Zero-Crossing Rate (as double)
    double zcr = zeroCrossCount(GetThresholdReal, INPUT_LENGTH);

    // Print final result
    printf("%f\n", zcr);

    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(signal);
    free(noise);
    free(noisy_sig);
    free(GetThresholdReal);

    return 0;
}
