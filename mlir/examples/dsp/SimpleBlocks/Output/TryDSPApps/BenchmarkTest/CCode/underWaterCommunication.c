#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100000000
#define FILTER_ORDER 5

// Function prototypes
double* getRangeOfVector(double start, int length, double increment);
void gain(double* output, const double* input, double multiplier, int length);
void sine(double* output, const double* input, int length);
void delay(double* output, const double* input, int delaySamples, int length);
void add(double* output, const double* input1, const double* input2, int length);
double* lowPassFIRFilter(double wc, int length);
double* hamming(int length);
void FIRFilterResponse(double* output, const double* input, const double* filter, int inputLength, int filterLength);
void thresholdUp(double* output, const double* input, double threshold, double defaultValue, int length);

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

double sinc(double x) {
    if (x == 0) return 1.0;
    return sin(x) / x;
}

double* lowPassFIRFilter(double wc, int length) {
    double* filter = malloc(length * sizeof(double));
    if (!filter) {
        perror("Memory allocation failed in lowPassFIRFilter");
        exit(EXIT_FAILURE);
    }
    int mid = (length - 1) / 2;
    for (int n = 0; n < length; n++) {
        if (n == mid) {
            filter[n] = wc / PI;
        } else {
            filter[n] = sinc(wc * (n - mid)) * wc / PI;
        }
    }
    return filter;
}

double* hamming(int length) {
    double* window = malloc(length * sizeof(double));
    if (!window) {
        perror("Memory allocation failed in hamming");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (length - 1));
    }
    return window;
}

void FIRFilterResponse(double* output, const double* input, const double* filter, int inputLength, int filterLength) {
    for (int i = 0; i < inputLength; i++) {
        output[i] = 0;
        for (int j = 0; j < filterLength; j++) {
            if (i - j >= 0) {
                output[i] += input[i - j] * filter[j];
            }
        }
    }
}

void thresholdUp(double* output, const double* input, double threshold, double defaultValue, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = (input[i] >= threshold) ? input[i] : defaultValue;
    }
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
    
    double fc = 1000;
    double wc = 2 * PI * fc / 500;  // wc should vary from 0 to pi
    
    double* lpf = lowPassFIRFilter(wc, FILTER_ORDER);
    double* hamming_window = hamming(FILTER_ORDER);
    
    double* lpf_w = malloc(FILTER_ORDER * sizeof(double));
    for (int i = 0; i < FILTER_ORDER; i++) {
        lpf_w[i] = lpf[i] * hamming_window[i];
    }
    
    double* FIRfilterResponse = malloc(INPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponse, noisy_sig, lpf_w, INPUT_LENGTH, FILTER_ORDER);
    
    double threshold = 0.5;
    double* GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));
    thresholdUp(GetThresholdReal, FIRfilterResponse, threshold, 0, INPUT_LENGTH);
    
    for (int i = 0; i < INPUT_LENGTH; i++) {
        printf("%f ", GetThresholdReal[i]);
    }
    printf("\n");
    
    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(signal);
    free(noise);
    free(noisy_sig);
    free(lpf);
    free(hamming_window);
    free(lpf_w);
    free(FIRfilterResponse);
    free(GetThresholdReal);
    
    return 0;
}