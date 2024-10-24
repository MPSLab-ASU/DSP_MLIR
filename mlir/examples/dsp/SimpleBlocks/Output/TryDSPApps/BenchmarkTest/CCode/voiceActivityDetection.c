#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100000000

double* getRangeOfVector(double start, int length, double increment);
void gain(double* output, const double* input, double multiplier, int length);
void sine(double* output, const double* input, int length);
void delay(double* output, const double* input, int delaySamples, int length);
void add(double* output, const double* input1, const double* input2, int length);
void threshold(double* output, const double* input, double thresholdValue, int length);
int zeroCrossCount(const double* input, int length);

// Function implementations
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

void gain(double* output, const double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

void sine(double* output, const double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = sin(input[i]);
    }
}

void delay(double* output, const double* input, int delaySamples, int length) {
    for (int i = 0; i < length; i++) {
        if (i < delaySamples) {
            output[i] = 0;
        } else {
            output[i] = input[i - delaySamples];
        }
    }
}

void add(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void threshold(double* output, const double* input, double thresholdValue, int length) {
    for (int i = 0; i < length; i++) {
        if (input[i] <= -thresholdValue || input[i] >= thresholdValue) {
            output[i] = input[i];
        } else {
            output[i] = 0;
        }
    }
}

int zeroCrossCount(const double* input, int length) {
    int count = 0;
    for (int i = 1; i < length; i++) {
        if ((input[i-1] > 0 && input[i] <= 0) || (input[i-1] < 0 && input[i] >= 0)) {
            count++;
        }
    }
    return count;
}

int main() {
    int fs = 1000;
    double* input = getRangeOfVector(0, INPUT_LENGTH, 1);
    
    double getMultiplier = 2 * PI * 5;
    double* getSinDuration = malloc(INPUT_LENGTH * sizeof(double));
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);
    
    double* signal = malloc(INPUT_LENGTH * sizeof(double));
    sine(signal, getSinDuration, INPUT_LENGTH);
    
    double* noise = malloc(INPUT_LENGTH * sizeof(double));
    delay(noise, signal, 5, INPUT_LENGTH);
    
    double* noisy_sig = malloc(INPUT_LENGTH * sizeof(double));
    add(noisy_sig, signal, noise, INPUT_LENGTH);
    
    double threshold_value = 0.8;
    double* GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));
    threshold(GetThresholdReal, noisy_sig, threshold_value, INPUT_LENGTH);
    
    int zcr = zeroCrossCount(GetThresholdReal, INPUT_LENGTH);
    
    for (int i = 0; i < INPUT_LENGTH; i++) {
        printf("%f ", GetThresholdReal[i]);
    }
    printf("\n");
    
    // Print zero-crossing count
    printf("Zero-crossing count: %d\n", zcr);
    
    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(signal);
    free(noise);
    free(noisy_sig);
    free(GetThresholdReal);
    
    return 0;
}