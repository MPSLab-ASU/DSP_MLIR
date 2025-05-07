#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define FS 1000
#define INPUT_LENGTH 50 
#define FILTER_ORDER 5

// Function prototypes
void getRangeOfVector(double* vector, double start, int length, double increment);
void gain(double* output, const double* input, double multiplier, int length);
void sine(double* output, const double* input, int length);
void delay(double* output, const double* input, int delaySamples, int length);
void add(double* output, const double* input1, const double* input2, int length);
double lowPassFIRFilter(double wc, int length);
void hamming(double* window, int length);
void FIRFilterResponse(double* output, double* input, double filter, int input_length);
void thresholdUp(double* output, const double* input, double threshold, double defaultValue, int length);
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

double sinc(double x) {
    return (fabs(x) < 1e-8) ? 1.0 : sin(x) / x;  // Handle division by zero
}

double lowPassFIRFilter(double wc, int length) {
    if (length == 1) {
        return wc / PI;  
    }
    return 0.0;  
}

void hamming(double* window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (length - 1));
    }
}

void FIRFilterResponse(double* output, double* input, double filter, int input_length) {
    for (int i = 0; i < input_length; i++) {
        output[i] = input[i] * filter;  // Element-wise multiplication with single value
    }
}

void thresholdUp(double* output, const double* input, double threshold, double defaultValue, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = (input[i] >= threshold) ? 1 : defaultValue;
    }
}

double getElemAtIndx(double* input, int index) {
    return input[index];
}

int main() {
    double pi = PI;

    double *input = malloc(INPUT_LENGTH * sizeof(double));
    double *getSinDuration = malloc(INPUT_LENGTH * sizeof(double));
    double *signal = malloc(INPUT_LENGTH * sizeof(double));
    double *noise = malloc(INPUT_LENGTH * sizeof(double));
    double *noisy_sig = malloc(INPUT_LENGTH * sizeof(double));
    double *FIRfilterResponseArray = malloc(INPUT_LENGTH * sizeof(double));
    double *GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));

    // Always check for successful allocation
    if (!input || !getSinDuration || !signal || !noise || !noisy_sig || 
        !FIRfilterResponseArray || !GetThresholdReal) {
        printf("Memory allocation failed\n");
        return -1;
    }

    getRangeOfVector(input, 0, INPUT_LENGTH, 0.000125);

    double getMultiplier = 2 * pi * 5;
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);
    sine(signal, getSinDuration, INPUT_LENGTH);
    delay(noise, signal, 5, INPUT_LENGTH);
    add(noisy_sig, signal, noise, INPUT_LENGTH);

    double wc = 2 * pi * 1000 / 500;
    int N = 5;
    double lpf = lowPassFIRFilter(wc, 1);  
    double hamming_window[N];
    hamming(hamming_window, N);
    double lpf_w = lpf * hamming_window[0];  
    FIRFilterResponse(FIRfilterResponseArray, noisy_sig, lpf_w, INPUT_LENGTH);
    double threshold = 0.05;
    thresholdUp(GetThresholdReal, FIRfilterResponseArray, threshold, 0, INPUT_LENGTH);

    double final1 = getElemAtIndx(GetThresholdReal, 3);
    printf("%f\n", final1);

    free(input);
    free(getSinDuration);
    free(signal);
    free(noise);
    free(noisy_sig);
    free(FIRfilterResponseArray);
    free(GetThresholdReal);

    return 0;
}

