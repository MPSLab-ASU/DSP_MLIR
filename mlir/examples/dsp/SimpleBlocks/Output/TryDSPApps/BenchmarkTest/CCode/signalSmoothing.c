#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 20000
#define SAMPLE_RATE 8000
#define TIME_INCREMENT 0.000125
#define WINDOW_SIZE 3

// Function declarations
void getRangeOfVector(double* vector, double start, int length, double increment);
void gain(double* output, double* input, double multiplier, int length);
void sine(double* output, double* input, int length);
void sliding_median_filter(double* input, double* output, int length);
void sliding_avg_filter(double* input, double* output, int length);
double min_of_three(double a, double b, double c);
double max_of_three(double a, double b, double c);

int main() {
    double fs = SAMPLE_RATE;
    double* input = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (input == NULL) {
        fprintf(stderr, "Memory allocation failed for input\n");
        return 1;
    }

    double f_sig = 500;
    double getMultiplier = 2 * PI * f_sig;
    
    getRangeOfVector(input, 0, INPUT_LENGTH, TIME_INCREMENT);
    
    double* getSinDuration = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (getSinDuration == NULL) {
        fprintf(stderr, "Memory allocation failed for getSinDuration\n");
        free(input);
        return 1;
    }
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);
    
    double* clean_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (clean_sig == NULL) {
        fprintf(stderr, "Memory allocation failed for clean_sig\n");
        free(input);
        free(getSinDuration);
        return 1;
    }
    sine(clean_sig, getSinDuration, INPUT_LENGTH);
    
    double f_noise = 3000;
    double* getNoiseSinDuration = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (getNoiseSinDuration == NULL) {
        fprintf(stderr, "Memory allocation failed for getNoiseSinDuration\n");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        return 1;
    }
    gain(getNoiseSinDuration, input, 2 * PI * f_noise, INPUT_LENGTH);
    
    double* noise = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (noise == NULL) {
        fprintf(stderr, "Memory allocation failed for noise\n");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        free(getNoiseSinDuration);
        return 1;
    }
    sine(noise, getNoiseSinDuration, INPUT_LENGTH);
    
    double* noise1 = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (noise1 == NULL) {
        fprintf(stderr, "Memory allocation failed for noise1\n");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        free(getNoiseSinDuration);
        free(noise);
        return 1;
    }
    gain(noise1, noise, 0.5, INPUT_LENGTH);
    
    double* noisy_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    if (noisy_sig == NULL) {
        fprintf(stderr, "Memory allocation failed for noisy_sig\n");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        free(getNoiseSinDuration);
        free(noise);
        free(noise1);
        return 1;
    }
    for (int i = 0; i < INPUT_LENGTH; i++) {
        noisy_sig[i] = clean_sig[i] + noise1[i];
    }
    
    double* median = (double*)malloc((INPUT_LENGTH - WINDOW_SIZE + 1) * sizeof(double));
    if (median == NULL) {
        fprintf(stderr, "Memory allocation failed for median\n");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        free(getNoiseSinDuration);
        free(noise);
        free(noise1);
        free(noisy_sig);
        return 1;
    }
    sliding_median_filter(noisy_sig, median, INPUT_LENGTH);
    
    double* average = (double*)malloc((INPUT_LENGTH - WINDOW_SIZE + 1) * sizeof(double));
    if (average == NULL) {
        fprintf(stderr, "Memory allocation failed for average\n");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        free(getNoiseSinDuration);
        free(noise);
        free(noise1);
        free(noisy_sig);
        free(median);
        return 1;
    }
    sliding_avg_filter(median, average, INPUT_LENGTH - WINDOW_SIZE + 1);
    
    printf("%f\n", average[3]); 
    
    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(clean_sig);
    free(getNoiseSinDuration);
    free(noise);
    free(noise1);
    free(noisy_sig);
    free(median);
    free(average);
    
    return 0;
}

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

// Function to find the minimum of three values
double min_of_three(double a, double b, double c) {
    double min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}

// Function to find the maximum of three values
double max_of_three(double a, double b, double c) {
    double max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}

// Function to apply sliding window average filter with kernel size of 3
void sliding_avg_filter(double* input, double* output, int length) {
    int new_length = length - WINDOW_SIZE + 1;
    for (int i = 0; i < new_length; i++) {
        output[i] = (input[i] + input[i + 1] + input[i + 2]) / 3.0;
    }
}

// Function to apply sliding window median filter with kernel size of 3
void sliding_median_filter(double* input, double* output, int length) {
    int new_length = length - WINDOW_SIZE + 1;
    for (int i = 0; i < new_length; i++) {
        double a = input[i];
        double b = input[i + 1];
        double c = input[i + 2];
        // Median formula: median = a + b + c - max(a, b, c) - min(a, b, c)
        double max_val = max_of_three(a, b, c);
        double min_val = min_of_three(a, b, c);
        output[i] = a + b + c - max_val - min_val;
    }
}