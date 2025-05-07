#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define FS 1000
#define INPUT_LENGTH 50000000
#define FILTER_SIZE 20
#define MAX_PEAKS 50

// Function prototypes
void getRangeOfVector(double* vector, double start, int length, double increment);
void gain(double* output, const double* input, double multiplier, int length);
void sine(double* output, const double* input, int length);
void delay(double* output, const double* input, int delaySamples, int length);
void add(double* output, const double* input1, const double* input2, int length);
void lmsFilterResponse(double* y, double* noisy_sig, double* clean_sig, double mu, int filterSize, int length);
void find_peaks(double* peaks, double* input, int length, double height, int distance);
double getElemAtIndx(double* input, int index);

// Function implementations
void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
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
        output[i] = (i < delaySamples) ? 0.0 : input[i - delaySamples];
    }
}

void add(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void lmsFilterResponse(double* y, double* noisy_sig, double* clean_sig, double mu, int filterSize, int length) {
    double w[FILTER_SIZE] = {0}; // Initialize weights to zero
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

void find_peaks(double* peaks, double* input, int length, double height, int distance) {
    int peakCount = 0;

    // Initialize peaks array with -1 (default no peaks)
    for (int i = 0; i < MAX_PEAKS; i++) {
        peaks[i] = -1;
    }

    for (int i = 1; i < length - 1; i++) {
        if (input[i] > input[i - 1] && input[i] > input[i + 1] && input[i] >= height) {
            // If it's the first peak, store it
            if (peakCount == 0) {
                peaks[peakCount++] = i;
            } else {
                // Ensure distance between peaks
                if (i - peaks[peakCount - 1] >= distance) {
                    peaks[peakCount++] = i;
                }
            }

            // Stop if max peaks reached
            if (peakCount >= MAX_PEAKS - 1) {
                break;
            }
        }
    }

    // Store peak count at the last index
    peaks[MAX_PEAKS - 1] = peakCount;
}

double getElemAtIndx(double* input, int index) {
    return input[index];
}

int main() {
    double pi = PI;
    int len = INPUT_LENGTH;

    // Dynamically allocate memory
    double *input = malloc(len * sizeof(double));
    double *getSinDuration = malloc(len * sizeof(double));
    double *sig1 = malloc(len * sizeof(double));
    double *getSinDuration2 = malloc(len * sizeof(double));
    double *sinsig2 = malloc(len * sizeof(double));
    double *sig2 = malloc(len * sizeof(double));
    double *signal = malloc(len * sizeof(double));
    double *noise = malloc(len * sizeof(double));
    double *noisy_sig = malloc(len * sizeof(double));
    double *y = malloc(len * sizeof(double));

    // Check allocation
    if (!input || !getSinDuration || !sig1 || !getSinDuration2 || !sinsig2 ||
        !sig2 || !signal || !noise || !noisy_sig || !y) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    getRangeOfVector(input, 0, len, 0.000125);
    double getMultiplier = 2 * pi * 10;
    gain(getSinDuration, input, getMultiplier, len);
    sine(sig1, getSinDuration, len);

    double getMultiplier2 = 2 * pi * 20;
    gain(getSinDuration2, input, getMultiplier2, len);
    sine(sinsig2, getSinDuration2, len);
    gain(sig2, sinsig2, 0.5, len);
    add(signal, sig1, sig2, len);
    delay(noise, signal, 5, len);
    add(noisy_sig, signal, noise, len);

    double mu = 0.01;
    lmsFilterResponse(y, noisy_sig, signal, mu, FILTER_SIZE, len);

    double peaks[MAX_PEAKS];
    find_peaks(peaks, y, len, 1.0, 50);

    double final1 = getElemAtIndx(peaks, 1);
    double final2 = getElemAtIndx(peaks, 0);
    printf("%f\t%f\n", final1, final2);

    // Free memory
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

