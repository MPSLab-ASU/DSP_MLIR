#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100000000
#define MAX_PEAKS 100  

// Function declarations
void getRangeOfVector(double* vector, double start, int length, double increment);
void gain(double* output, double* input, double multiplier, int length);
void sine(double* output, double* input, int length);
void delay(double* output, double* input, int length, int delay);
void add_signals(double* output, double* input1, double* input2, int length);
void lmsFilterResponse(double* output, double* input, double* desired, double mu, int filterSize, int length);
int find_peaks(double* signal, int length, double threshold, int minDistance, int* peaks);

int main() {
    double fs = 1000;
    double* input = (double*)malloc(INPUT_LENGTH * sizeof(double));
    getRangeOfVector(input, 0, INPUT_LENGTH, 0.000125);

    double getMultiplier = 2 * PI * 10;
    double* getSinDuration = (double*)malloc(INPUT_LENGTH * sizeof(double));
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);

    double* sig1 = (double*)malloc(INPUT_LENGTH * sizeof(double));
    sine(sig1, getSinDuration, INPUT_LENGTH);

    double getMultiplier2 = 2 * PI * 20;
    double* getSinDuration2 = (double*)malloc(INPUT_LENGTH * sizeof(double));
    gain(getSinDuration2, input, getMultiplier2, INPUT_LENGTH);

    double* sinsig2 = (double*)malloc(INPUT_LENGTH * sizeof(double));
    sine(sinsig2, getSinDuration2, INPUT_LENGTH);

    double* sig2 = (double*)malloc(INPUT_LENGTH * sizeof(double));
    gain(sig2, sinsig2, 0.5, INPUT_LENGTH);

    double* signal = (double*)malloc(INPUT_LENGTH * sizeof(double));
    add_signals(signal, sig1, sig2, INPUT_LENGTH);

    double* noise = (double*)malloc(INPUT_LENGTH * sizeof(double));
    delay(noise, signal, INPUT_LENGTH, 5);

    double* noisy_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    add_signals(noisy_sig, signal, noise, INPUT_LENGTH);

    double mu = 0.01;
    int filterSize = 20;
    double* y = (double*)malloc(INPUT_LENGTH * sizeof(double));
    lmsFilterResponse(y, noisy_sig, signal, mu, filterSize, INPUT_LENGTH);

    int peaks[MAX_PEAKS];
    int num_peaks = find_peaks(signal, INPUT_LENGTH, 1, 50, peaks);

    
    printf("%d %d", peaks[1], peaks[2]);
    
    printf("\n");

    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(sig1);
    free(getSinDuration2);
    free(sinsig2);
    free(sig2);
    free(signal);
    free(noise);
    free(noisy_sig);
    free(y);

    return 0;
}

// Function implementations

void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
}

void gain(double* output, double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

void sine(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = sin(input[i]);
    }
}

void delay(double* output, double* input, int length, int delay) {
    for (int i = 0; i < length; i++) {
        if (i < delay) {
            output[i] = 0;
        } else {
            output[i] = input[i - delay];
        }
    }
}

void add_signals(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
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
        output[n] = e;
    }
}

int find_peaks(double* signal, int length, double threshold, int minDistance, int* peaks) {
    int num_peaks = 0;  
    for (int i = 1; i < length - 1 && num_peaks < MAX_PEAKS; i++) {
        if (signal[i] > threshold && signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
            peaks[num_peaks++] = i;
            i += minDistance;  
        }
    }
    return num_peaks;
}