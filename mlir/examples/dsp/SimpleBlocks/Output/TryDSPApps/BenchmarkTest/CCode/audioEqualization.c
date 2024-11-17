#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100000000

// Function prototypes
double* getRangeOfVector(double start, int length, double increment);
double* lowPassFIRFilter(double wc, int length);
double* highPassFIRFilter(double wc, int length);
double* hamming(int length);
void elementWiseMultiplication(double* output, const double* array1, const double* array2, int length);
void FIRFilterResponse(double* output, const double* input, const double* filter, int inputLength, int filterLength);
void gain(double* output, const double* input, double gainFactor, int length);
void add(double* output, const double* input1, const double* input2, int length);
void sub(double* output, const double* input1, const double* input2, int length);

// Implement the provided functions
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
            filter[n] = sin(wc * (n - mid)) / (PI * (n - mid));
        }
    }
    return filter;
}

void elementWiseMultiplication(double* output, const double* array1, const double* array2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = array1[i] * array2[i];
    }
}

// Implement additional required functions
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

double* highPassFIRFilter(double wc, int length) {
    double* lpf = lowPassFIRFilter(wc, length);
    double* hpf = malloc(length * sizeof(double));
    if (!hpf) {
        perror("Memory allocation failed in highPassFIRFilter");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        hpf[i] = -lpf[i];
    }
    int mid = (length - 1) / 2;
    hpf[mid] += 1.0;
    free(lpf);
    return hpf;
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

void gain(double* output, const double* input, double gainFactor, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * gainFactor;
    }
}

void add(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void sub(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] - input2[i];
    }
}

int main() {
    double* input = getRangeOfVector(0, INPUT_LENGTH, 1);
    double pi = PI;
    double fc = 300;
    double Fs = 8000;
    double gainForBass = 2;
    double gainForMid = 1.5;
    double gainForTreble = 0.8;

    double wc = 2 * pi * fc / Fs;
    int N = 101;

    // Low-pass filter
    double* lpf = lowPassFIRFilter(wc, N);
    double* hamming_window = hamming(N);
    double* lpf_w = malloc(N * sizeof(double));
    elementWiseMultiplication(lpf_w, lpf, hamming_window, N);

    double* FIRfilterResponseForLpf = malloc(INPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForLpf, input, lpf_w, INPUT_LENGTH, N);

    double* gainWithLpf = malloc(INPUT_LENGTH * sizeof(double));
    gain(gainWithLpf, FIRfilterResponseForLpf, gainForBass, INPUT_LENGTH);

    // High-pass filter
    double fc2 = 1500;
    double wc2 = 2 * pi * fc2 / Fs;
    double* hpf = highPassFIRFilter(wc2, N);
    double* hpf_w = malloc(N * sizeof(double));
    elementWiseMultiplication(hpf_w, hpf, hamming_window, N);

    double* FIRfilterResponseForHpf = malloc(INPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForHpf, input, hpf_w, INPUT_LENGTH, N);

    double* gainWithHpf = malloc(INPUT_LENGTH * sizeof(double));
    gain(gainWithHpf, FIRfilterResponseForHpf, gainForTreble, INPUT_LENGTH);

    // Band-pass filter
    double* lpf2 = lowPassFIRFilter(wc2, N);
    double* lpf2_w = malloc(N * sizeof(double));
    elementWiseMultiplication(lpf2_w, lpf2, hamming_window, N);

    double* bpf_w = malloc(N * sizeof(double));
    sub(bpf_w, lpf2_w, lpf_w, N);

    double* FIRfilterResponseForBpf = malloc(INPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForBpf, input, bpf_w, INPUT_LENGTH, N);

    double* gainWithBpf = malloc(INPUT_LENGTH * sizeof(double));
    gain(gainWithBpf, FIRfilterResponseForBpf, gainForMid, INPUT_LENGTH);

    // Final audio
    double* final_audio = malloc(INPUT_LENGTH * sizeof(double));
    add(final_audio, gainWithLpf, gainWithHpf, INPUT_LENGTH);
    add(final_audio, final_audio, gainWithBpf, INPUT_LENGTH);

    // Print results
    printf("Element at index 3: %f\n", final_audio[3]);
    for (int i = 0; i < INPUT_LENGTH; i++) {
        printf("%f ", final_audio[i]);
    }
    printf("\n");

    // Free allocated memory
    free(input);
    free(lpf);
    free(hamming_window);
    free(lpf_w);
    free(FIRfilterResponseForLpf);
    free(gainWithLpf);
    free(hpf);
    free(hpf_w);
    free(FIRfilterResponseForHpf);
    free(gainWithHpf);
    free(lpf2);
    free(lpf2_w);
    free(bpf_w);
    free(FIRfilterResponseForBpf);
    free(gainWithBpf);
    free(final_audio);

    return 0;
}