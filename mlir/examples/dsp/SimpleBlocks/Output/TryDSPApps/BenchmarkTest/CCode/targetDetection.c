#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define FS 1000
#define INPUT_LENGTH 1000
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
    double input[INPUT_LENGTH];
    getRangeOfVector(input, 0, INPUT_LENGTH, 0.000125);

    double getMultiplier = 2 * pi * 10;
    double getSinDuration[INPUT_LENGTH];
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);

    double sig1[INPUT_LENGTH];
    sine(sig1, getSinDuration, INPUT_LENGTH);

    double getMultiplier2 = 2 * pi * 20;
    double getSinDuration2[INPUT_LENGTH];
    gain(getSinDuration2, input, getMultiplier2, INPUT_LENGTH);

    double sinsig2[INPUT_LENGTH];
    sine(sinsig2, getSinDuration2, INPUT_LENGTH);

    double sig2[INPUT_LENGTH];
    gain(sig2, sinsig2, 0.5, INPUT_LENGTH);

    double signal[INPUT_LENGTH];
    add(signal, sig1, sig2, INPUT_LENGTH);

    double noise[INPUT_LENGTH];
    delay(noise, signal, 5, INPUT_LENGTH);

    double noisy_sig[INPUT_LENGTH];
    add(noisy_sig, signal, noise, INPUT_LENGTH);

    double mu = 0.01;
    double y[INPUT_LENGTH];
    lmsFilterResponse(y, noisy_sig, signal, mu, FILTER_SIZE, INPUT_LENGTH);

    double peaks[MAX_PEAKS];
    find_peaks(peaks, y, INPUT_LENGTH, 1.0, 50);

    double final1 = getElemAtIndx(peaks, 1);
    double final2 = getElemAtIndx(peaks, 0);

    printf("%f\t", final1);
    printf("%f", final2);
   

    return 0;
}
